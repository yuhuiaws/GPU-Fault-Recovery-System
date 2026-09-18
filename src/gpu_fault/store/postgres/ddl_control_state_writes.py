"""Database-routed writes keep cutover atomic with in-flight legacy writers."""

from __future__ import annotations

from typing import Any

from gpu_fault.store.postgres.ddl_helpers import _ensure_trigger


def create_native_state_writer(cursor: Any) -> None:
    cursor.execute(
        """
        CREATE OR REPLACE FUNCTION gpu_fault_put_native_control_state(
            record_kind TEXT, record_key TEXT, columns JSONB,
            expected JSONB DEFAULT NULL, guard_versions BOOLEAN DEFAULT FALSE
        ) RETURNS BOOLEAN LANGUAGE plpgsql AS $$
        DECLARE descriptor RECORD; assignments TEXT; statement TEXT; affected BIGINT;
                input_name TEXT; condition TEXT;
        BEGIN
            SELECT * INTO STRICT descriptor FROM gpu_fault_control_state_descriptor(record_kind);
            IF columns->>descriptor.key_field IS DISTINCT FROM record_key THEN
                RAISE EXCEPTION 'control-state record identity differs' USING ERRCODE='55000';
            END IF;
            input_name := CASE WHEN expected IS NULL THEN 'excluded' ELSE 'candidate' END;
            SELECT string_agg(format(
                '%1$I=CASE WHEN target.%1$I IS NOT DISTINCT FROM %2$I.%1$I '
                'THEN target.%1$I ELSE %2$I.%1$I END', attname, input_name
            ), ', ' ORDER BY attnum) INTO assignments
            FROM pg_attribute
            WHERE attrelid=to_regclass(descriptor.table_name)
              AND attnum>0 AND NOT attisdropped AND attname<>descriptor.key_field;
            IF assignments IS NULL THEN
                RAISE EXCEPTION 'control-state table is unavailable' USING ERRCODE='55000';
            END IF;
            IF expected IS NOT NULL THEN
                statement := format(
                    'UPDATE %1$I target SET %2$s '
                    'FROM jsonb_populate_record(NULL::%1$I, $1) candidate '
                    'WHERE target.%3$I=$2 AND %4$I(target)=$3',
                    descriptor.table_name, assignments, descriptor.key_field,
                    descriptor.payload_function
                );
                EXECUTE statement USING columns, record_key, expected;
            ELSE
                condition := '';
                IF guard_versions THEN
                    IF record_kind<>'workflow' THEN
                        RAISE EXCEPTION 'workflow version guard used for another kind';
                    END IF;
                    condition := ' WHERE target.merge_revision=excluded.merge_revision '
                        'AND target.execution_epoch=excluded.execution_epoch '
                        'AND target.fencing_token=excluded.fencing_token';
                END IF;
                statement := format(
                    'INSERT INTO %1$I AS target '
                    'SELECT * FROM jsonb_populate_record(NULL::%1$I, $1) '
                    'ON CONFLICT(%2$I) DO UPDATE SET %3$s%4$s',
                    descriptor.table_name, descriptor.key_field, assignments, condition
                );
                EXECUTE statement USING columns;
            END IF;
            GET DIAGNOSTICS affected=ROW_COUNT;
            RETURN affected=1;
        END
        $$
        """
    )


def create_control_state_writer(cursor: Any) -> None:
    cursor.execute(
        """
        CREATE OR REPLACE FUNCTION gpu_fault_put_control_state(
            record_kind TEXT, record_key TEXT, columns JSONB,
            expected JSONB DEFAULT NULL, guard_versions BOOLEAN DEFAULT FALSE
        ) RETURNS BOOLEAN LANGUAGE plpgsql AS $$
        DECLARE current_mode TEXT; descriptor RECORD; payload_value JSONB; affected BIGINT;
        BEGIN
            current_mode := gpu_fault_control_state_lock_mode(record_kind);
            SELECT * INTO STRICT descriptor FROM gpu_fault_control_state_descriptor(record_kind);
            IF columns->>descriptor.key_field IS DISTINCT FROM record_key THEN
                RAISE EXCEPTION 'control-state record identity differs' USING ERRCODE='55000';
            END IF;
            IF current_mode='dedicated' THEN
                RETURN gpu_fault_put_native_control_state(
                    record_kind, record_key, columns, expected, guard_versions
                );
            END IF;
            EXECUTE format(
                'SELECT %I(jsonb_populate_record(NULL::%I, $1))',
                descriptor.payload_function, descriptor.table_name
            ) INTO payload_value USING columns;
            IF expected IS NOT NULL THEN
                UPDATE gpu_fault_objects SET payload=payload_value
                WHERE kind=record_kind AND key=record_key AND payload=expected;
            ELSIF guard_versions THEN
                IF record_kind<>'workflow' THEN
                    RAISE EXCEPTION 'workflow version guard used for another kind';
                END IF;
                INSERT INTO gpu_fault_objects AS target(kind, key, payload)
                VALUES (record_kind, record_key, payload_value)
                ON CONFLICT(kind, key) DO UPDATE SET payload=excluded.payload
                WHERE coalesce(target.payload->>'merge_revision', '0')
                          =coalesce(excluded.payload->>'merge_revision', '0')
                  AND coalesce(target.payload->>'execution_epoch', '0')
                          =coalesce(excluded.payload->>'execution_epoch', '0')
                  AND target.payload->>'fencing_token'=excluded.payload->>'fencing_token';
            ELSE
                INSERT INTO gpu_fault_objects(kind, key, payload)
                VALUES (record_kind, record_key, payload_value)
                ON CONFLICT(kind, key) DO UPDATE SET payload=excluded.payload;
            END IF;
            GET DIAGNOSTICS affected=ROW_COUNT;
            RETURN affected=1;
        END
        $$
        """
    )


def create_control_state_mirror(cursor: Any) -> None:
    cursor.execute(
        """
        CREATE OR REPLACE FUNCTION gpu_fault_objects_control_state_mirror()
        RETURNS trigger LANGUAGE plpgsql AS $$
        DECLARE record_kind TEXT; record_key TEXT; descriptor RECORD; columns JSONB;
        BEGIN
            record_kind := CASE WHEN TG_OP='DELETE' THEN OLD.kind ELSE NEW.kind END;
            IF record_kind NOT IN ('remote_command', 'workflow') THEN RETURN NULL; END IF;
            IF gpu_fault_control_state_lock_mode(record_kind)<>'dual' THEN RETURN NULL; END IF;
            record_key := CASE WHEN TG_OP='DELETE' THEN OLD.key ELSE NEW.key END;
            SELECT * INTO STRICT descriptor FROM gpu_fault_control_state_descriptor(record_kind);
            IF TG_OP='DELETE' THEN
                EXECUTE format('DELETE FROM %I WHERE %I=$1',
                    descriptor.table_name, descriptor.key_field) USING record_key;
            ELSE
                EXECUTE format('SELECT to_jsonb(%I($1))', descriptor.columns_function)
                    INTO columns USING NEW.payload;
                PERFORM gpu_fault_put_native_control_state(record_kind, record_key, columns);
            END IF;
            RETURN NULL;
        END
        $$
        """
    )
    _ensure_trigger(
        cursor,
        "gpu_fault_objects_control_state_mirror_trigger",
        "gpu_fault_objects",
        """
        CREATE TRIGGER gpu_fault_objects_control_state_mirror_trigger
        AFTER INSERT OR DELETE OR UPDATE ON gpu_fault_objects
        FOR EACH ROW EXECUTE FUNCTION gpu_fault_objects_control_state_mirror()
        """,
    )


def create_control_state_patch_writer(cursor: Any) -> None:
    cursor.execute(
        """
        CREATE OR REPLACE FUNCTION gpu_fault_patch_control_state(
            record_kind TEXT, record_key TEXT, changed_columns JSONB
        ) RETURNS BOOLEAN LANGUAGE plpgsql AS $$
        DECLARE current_mode TEXT; descriptor RECORD; assignments TEXT; affected BIGINT;
                previous JSONB; columns JSONB; name TEXT;
        BEGIN
            current_mode := gpu_fault_control_state_lock_mode(record_kind);
            SELECT * INTO STRICT descriptor FROM gpu_fault_control_state_descriptor(record_kind);
            IF jsonb_typeof(changed_columns)<>'object' OR changed_columns='{}'::jsonb THEN
                RAISE EXCEPTION 'empty or invalid partial control-state update';
            END IF;
            FOR name IN SELECT jsonb_object_keys(changed_columns) LOOP
                IF name IN (descriptor.key_field, descriptor.snapshot_field, 'execution_owner')
                   OR NOT EXISTS (
                       SELECT 1 FROM pg_attribute
                       WHERE attrelid=to_regclass(descriptor.table_name)
                         AND attname=name AND attnum>0 AND NOT attisdropped
                   ) THEN
                    RAISE EXCEPTION 'invalid partial control-state column';
                END IF;
            END LOOP;
            IF current_mode<>'dedicated' THEN
                SELECT payload INTO previous FROM gpu_fault_objects
                WHERE kind=record_kind AND key=record_key FOR UPDATE;
                IF NOT FOUND THEN RETURN FALSE; END IF;
                EXECUTE format('SELECT to_jsonb(%I($1))', descriptor.columns_function)
                    INTO columns USING previous;
                RETURN gpu_fault_put_control_state(record_kind, record_key, columns || changed_columns);
            END IF;
            SELECT string_agg(format('%1$I=candidate.%1$I', key), ', ' ORDER BY key)
                INTO assignments FROM jsonb_object_keys(changed_columns) AS keys(key);
            EXECUTE format(
                'UPDATE %1$I target SET %2$s '
                'FROM jsonb_populate_record(NULL::%1$I, $1) candidate WHERE target.%3$I=$2',
                descriptor.table_name, assignments, descriptor.key_field
            ) USING changed_columns, record_key;
            GET DIAGNOSTICS affected=ROW_COUNT;
            RETURN affected=1;
        END
        $$
        """
    )


def create_control_state_lock_reader(cursor: Any) -> None:
    cursor.execute(
        """
        CREATE OR REPLACE FUNCTION gpu_fault_lock_control_state(record_kind TEXT, record_key TEXT)
        RETURNS TABLE(payload JSONB) LANGUAGE plpgsql AS $$
        DECLARE current_mode TEXT; descriptor RECORD;
        BEGIN
            current_mode := gpu_fault_control_state_lock_mode(record_kind);
            IF current_mode<>'dedicated' THEN
                RETURN QUERY SELECT o.payload FROM gpu_fault_objects o
                WHERE o.kind=record_kind AND o.key=record_key FOR UPDATE;
            ELSE
                SELECT * INTO STRICT descriptor FROM gpu_fault_control_state_descriptor(record_kind);
                RETURN QUERY EXECUTE format(
                    'SELECT %I(target) FROM %I target WHERE %I=$1 FOR UPDATE',
                    descriptor.payload_function, descriptor.table_name, descriptor.key_field
                ) USING record_key;
            END IF;
        END
        $$
        """
    )


def create_control_state_delete_writer(cursor: Any) -> None:
    cursor.execute(
        """
        CREATE OR REPLACE FUNCTION gpu_fault_delete_control_state(
            record_kind TEXT, record_key TEXT, expected JSONB DEFAULT NULL,
            skip_locked BOOLEAN DEFAULT FALSE
        ) RETURNS BOOLEAN LANGUAGE plpgsql AS $$
        DECLARE current_mode TEXT; descriptor RECORD; affected BIGINT;
                lock_clause TEXT; condition TEXT;
        BEGIN
            current_mode := CASE
                WHEN record_kind IN ('workflow', 'remote_command')
                THEN gpu_fault_control_state_lock_mode(record_kind)
                ELSE 'legacy' END;
            lock_clause := CASE WHEN skip_locked THEN ' SKIP LOCKED' ELSE '' END;
            IF current_mode<>'dedicated' THEN
                condition := '$3 IS NULL OR payload=$3';
                IF current_mode='dual' AND expected IS NOT NULL THEN
                    SELECT * INTO STRICT descriptor
                    FROM gpu_fault_control_state_descriptor(record_kind);
                    -- A dual read may reconstruct lifted defaults and timestamps.
                    -- Compare that complete projection without dropping a field.
                    condition := condition || format(' OR %I(%I(payload))=$3',
                        descriptor.payload_function, descriptor.columns_function);
                END IF;
                EXECUTE
                    'WITH victim AS (SELECT key FROM gpu_fault_objects '
                    'WHERE kind=$1 AND key=$2 AND (' || condition || ') '
                    'FOR UPDATE' || lock_clause || ') '
                    'DELETE FROM gpu_fault_objects t USING victim v '
                    'WHERE t.kind=$1 AND t.key=v.key'
                    USING record_kind, record_key, expected;
            ELSE
                SELECT * INTO STRICT descriptor FROM gpu_fault_control_state_descriptor(record_kind);
                EXECUTE format(
                    'WITH victim AS (SELECT %1$I FROM %2$I t '
                    'WHERE %1$I=$1 AND ($2 IS NULL OR %3$I(t)=$2) FOR UPDATE%4$s) '
                    'DELETE FROM %2$I t USING victim v WHERE t.%1$I=v.%1$I',
                    descriptor.key_field, descriptor.table_name,
                    descriptor.payload_function, lock_clause
                ) USING record_key, expected;
            END IF;
            GET DIAGNOSTICS affected=ROW_COUNT;
            RETURN affected=1;
        END
        $$
        """
    )
