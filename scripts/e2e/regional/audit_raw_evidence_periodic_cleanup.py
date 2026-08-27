from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import psycopg


def main() -> None:
    key = f"audit-expired-evidence-{uuid4()}"
    now = datetime.now(timezone.utc)
    payload = {
        "record_id": key,
        "cluster_id": "audit-cluster",
        "node_id": "audit-node",
        "kind": "WORKLOAD_LOG",
        "observed_at": (now - timedelta(hours=2)).isoformat(),
        "ingested_at": (now - timedelta(hours=2)).isoformat(),
        "expires_at": (now - timedelta(hours=1)).isoformat(),
        "attempt_ids": ["audit-attempt"],
        "payload": {"audit": True},
    }
    with psycopg.connect(os.environ["GPU_FAULT_STORE_URL"]) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO gpu_fault_objects(kind, key, payload)
                VALUES ('raw_evidence', %s, %s::jsonb)
                """,
                (key, json.dumps(payload)),
            )
        connection.commit()

        started = time.monotonic()
        try:
            for _ in range(30):
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT count(*)
                        FROM gpu_fault_objects
                        WHERE kind='raw_evidence' AND key=%s
                        """,
                        (key,),
                    )
                    remaining = cursor.fetchone()[0]
                if remaining == 0:
                    print(
                        "deleted",
                        key,
                        round(time.monotonic() - started, 3),
                    )
                    return
                time.sleep(5)
            raise RuntimeError(f"expired raw evidence was not deleted: {key}")
        finally:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    DELETE FROM gpu_fault_objects
                    WHERE kind='raw_evidence' AND key=%s
                    """,
                    (key,),
                )
            connection.commit()


if __name__ == "__main__":
    main()
