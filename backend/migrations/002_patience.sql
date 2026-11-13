-- Patience deadline for reneging trials; NULL means infinite patience (legacy rows keep no-abandonment semantics).
ALTER TABLE simulation_runs ADD COLUMN IF NOT EXISTS patience_minutes double precision;
