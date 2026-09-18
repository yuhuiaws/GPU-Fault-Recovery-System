from __future__ import annotations

AUDIT_PURGE_STATEMENTS = (
    (
        "gpu_fault_processor_queue",
        "DELETE FROM gpu_fault_processor_queue WHERE cluster_id LIKE %s",
        "cluster",
    ),
    (
        "gpu_fault_processor_lanes",
        "DELETE FROM gpu_fault_processor_lanes WHERE ordering_key LIKE %s",
        "cluster",
    ),
    (
        "gpu_fault_processor_queue_counts",
        "DELETE FROM gpu_fault_processor_queue_counts WHERE cluster_id LIKE %s",
        "cluster",
    ),
    (
        "gpu_fault_gpu_metric_latest",
        "DELETE FROM gpu_fault_gpu_metric_latest WHERE cluster_id LIKE %s",
        "cluster",
    ),
    (
        "gpu_fault_gpu_metrics_batches",
        "DELETE FROM gpu_fault_gpu_metrics_batches WHERE cluster_id LIKE %s",
        "cluster",
    ),
    (
        "gpu_fault_attempt_observations",
        "DELETE FROM gpu_fault_attempt_observations WHERE cluster_id LIKE %s",
        "cluster",
    ),
    (
        "gpu_fault_training_progress",
        "DELETE FROM gpu_fault_training_progress WHERE cluster_id LIKE %s",
        "cluster",
    ),
    (
        "gpu_fault_action_workflows",
        "WITH victims AS MATERIALIZED ("
        "SELECT key,payload FROM gpu_fault_control_records WHERE kind='workflow' AND key LIKE %s"
        ") SELECT key FROM victims WHERE gpu_fault_delete_control_state('workflow',key,payload)",
        "action_workflow",
    ),
    (
        "gpu_fault_notification_results",
        """
        DELETE FROM gpu_fault_objects AS result
        WHERE result.kind='notification_result'
          AND result.payload->>'notification_id' IN (
              SELECT notification.key
              FROM gpu_fault_objects AS notification
              WHERE notification.kind='notification'
                AND notification.payload->>'cluster_name' LIKE %s
          )
        """,
        "cluster",
    ),
    (
        "gpu_fault_notification_deliveries",
        """
        DELETE FROM gpu_fault_objects AS delivery
        WHERE delivery.kind='notification_delivery'
          AND delivery.payload->>'notification_id' IN (
              SELECT notification.key
              FROM gpu_fault_objects AS notification
              WHERE notification.kind='notification'
                AND notification.payload->>'cluster_name' LIKE %s
          )
        """,
        "cluster",
    ),
    (
        "gpu_fault_notifications",
        """
        DELETE FROM gpu_fault_objects
        WHERE kind='notification'
          AND payload->>'cluster_name' LIKE %s
        """,
        "cluster",
    ),
    (
        "gpu_fault_regional_clusters",
        "DELETE FROM gpu_fault_objects WHERE kind='regional_cluster' AND key LIKE %s",
        "cluster",
    ),
    (
        "gpu_fault_markers",
        """
        DELETE FROM gpu_fault_objects
        WHERE kind='marker'
          AND payload::text LIKE %s
        """,
        "cluster_contains",
    ),
    (
        "gpu_fault_links",
        """
        WITH pattern AS (
            SELECT rtrim(%s, '%%') AS prefix
        )
        DELETE FROM gpu_fault_links AS link
        USING pattern
        WHERE strpos(link.key, pattern.prefix) > 0
           OR strpos(link.value, pattern.prefix) > 0
        """,
        "cluster",
    ),
    (
        "gpu_fault_objects",
        "WITH victims AS MATERIALIZED ("
        "SELECT kind,key,payload FROM gpu_fault_control_records WHERE payload->>'cluster_id' LIKE %s"
        ") SELECT key FROM victims WHERE gpu_fault_delete_control_state(kind,key,payload)",
        "cluster",
    ),
)
