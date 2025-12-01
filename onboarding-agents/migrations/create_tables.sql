CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS runs (
  id uuid PRIMARY KEY,
  created_at timestamp with time zone DEFAULT now(),
  input_plan text,
  logs jsonb
);

CREATE TABLE IF NOT EXISTS tasks (
  id serial PRIMARY KEY,
  run_id uuid REFERENCES runs(id) ON DELETE CASCADE,
  title varchar NOT NULL,
  description text,
  due_date varchar,
  owner varchar,
  raw jsonb,
  created_at timestamp with time zone DEFAULT now()
);

CREATE TYPE task_status_enum AS ENUM ('on_track','delayed','at_risk');

CREATE TABLE IF NOT EXISTS status_analysis (
  id serial PRIMARY KEY,
  run_id uuid REFERENCES runs(id) ON DELETE CASCADE,
  task_id integer,
  status task_status_enum NOT NULL,
  explanation text,
  created_at timestamp with time zone DEFAULT now()
);

CREATE TABLE IF NOT EXISTS recommendations (
  id serial PRIMARY KEY,
  run_id uuid REFERENCES runs(id) ON DELETE CASCADE,
  note text,
  created_at timestamp with time zone DEFAULT now()
);
