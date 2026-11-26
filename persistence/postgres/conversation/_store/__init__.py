"""The PostgreSQL conversation store: the ported SQLite reference store, one PG dialect module per reference
file (messages/writes/reads/repair/rewind/prune_rewrite/coverage/stored_reads/timeline), over
``PostgresStore`` (``%s`` placeholders, BYTEA payload columns, the DDL display triggers)."""
