-- ============================================================
-- FL Platform — Supabase Schema + RLS
-- Run in Supabase SQL Editor
-- ============================================================

-- Enable UUID extension
create extension if not exists "pgcrypto";

-- ── Users ──────────────────────────────────────────────────
create table if not exists public.users (
  id          uuid primary key default gen_random_uuid(),
  email       text unique not null,
  role        text not null default 'user'
                check (role in ('super_admin', 'admin', 'user')),
  created_at  timestamptz default now()
);

alter table public.users enable row level security;

create policy "Users see own row"
  on public.users for select
  using (auth.uid() = id);

-- ── Datasets ───────────────────────────────────────────────
create table if not exists public.datasets (
  id            uuid primary key default gen_random_uuid(),
  user_id       uuid references public.users(id) on delete cascade,
  filename      text not null,
  storage_path  text not null,
  cols          jsonb,
  row_count     integer,
  created_at    timestamptz default now()
);

alter table public.datasets enable row level security;

create policy "Users see own datasets"
  on public.datasets for all
  using (auth.uid() = user_id);

-- ── Experiments ────────────────────────────────────────────
create table if not exists public.experiments (
  id            uuid primary key default gen_random_uuid(),
  user_id       uuid references public.users(id) on delete cascade,
  dataset_id    uuid references public.datasets(id),
  algorithm     text not null
                  check (algorithm in ('fedavg','fedprox','scaffold','dpsgd','central')),
  status        text not null default 'pending'
                  check (status in ('pending','running','completed','failed')),
  hyperparams   jsonb,
  celery_task_id text,
  created_at    timestamptz default now(),
  updated_at    timestamptz default now()
);

alter table public.experiments enable row level security;

create policy "Admin sees all experiments"
  on public.experiments for select
  using (
    exists (select 1 from public.users u where u.id = auth.uid() and u.role in ('admin','super_admin'))
    or auth.uid() = user_id
  );

create policy "Users manage own experiments"
  on public.experiments for insert
  with check (auth.uid() = user_id);

create policy "Users update own experiments"
  on public.experiments for update
  using (auth.uid() = user_id);

-- ── Rounds ─────────────────────────────────────────────────
create table if not exists public.rounds (
  id              uuid primary key default gen_random_uuid(),
  experiment_id   uuid references public.experiments(id) on delete cascade,
  round_num       integer not null,
  loss            float,
  accuracy        float,
  val_loss        float,
  val_accuracy    float,
  num_clients     integer,
  created_at      timestamptz default now()
);

alter table public.rounds enable row level security;

create policy "Rounds follow experiment RLS"
  on public.rounds for select
  using (
    exists (
      select 1 from public.experiments e
      where e.id = experiment_id
        and (e.user_id = auth.uid()
          or exists (select 1 from public.users u where u.id = auth.uid() and u.role in ('admin','super_admin')))
    )
  );

-- ── Models ─────────────────────────────────────────────────
create table if not exists public.models (
  id              uuid primary key default gen_random_uuid(),
  experiment_id   uuid references public.experiments(id) on delete cascade,
  weights_path    text,
  version         integer default 1,
  created_at      timestamptz default now()
);

alter table public.models enable row level security;

create policy "Models follow experiment RLS"
  on public.models for select
  using (
    exists (
      select 1 from public.experiments e
      where e.id = experiment_id
        and (e.user_id = auth.uid()
          or exists (select 1 from public.users u where u.id = auth.uid() and u.role in ('admin','super_admin')))
    )
  );

-- ── Metrics ────────────────────────────────────────────────
create table if not exists public.metrics (
  id              uuid primary key default gen_random_uuid(),
  experiment_id   uuid references public.experiments(id) on delete cascade,
  accuracy        float,
  f1              float,
  auc             float,
  precision_score float,
  recall          float,
  confusion_matrix jsonb,
  roc_curve       jsonb,
  feature_importance jsonb,
  created_at      timestamptz default now()
);

alter table public.metrics enable row level security;

-- Metrics are public read (no PII)
create policy "Public can read metrics"
  on public.metrics for select
  using (true);

-- ── Privacy Budget ─────────────────────────────────────────
create table if not exists public.privacy_budget (
  id              uuid primary key default gen_random_uuid(),
  experiment_id   uuid references public.experiments(id) on delete cascade,
  round_num       integer not null,
  epsilon         float not null,
  delta           float not null,
  noise_multiplier float,
  clip_norm       float,
  created_at      timestamptz default now()
);

alter table public.privacy_budget enable row level security;

create policy "Privacy budget follows experiment RLS"
  on public.privacy_budget for select
  using (
    exists (
      select 1 from public.experiments e
      where e.id = experiment_id
        and (e.user_id = auth.uid()
          or exists (select 1 from public.users u where u.id = auth.uid() and u.role in ('admin','super_admin')))
    )
  );

-- ── Predictions ────────────────────────────────────────────
create table if not exists public.predictions (
  id              uuid primary key default gen_random_uuid(),
  user_id         uuid references public.users(id) on delete cascade,
  model_id        uuid references public.models(id),
  input_hash      text,
  output          integer,
  confidence      float,
  batch_id        text,
  created_at      timestamptz default now()
);

alter table public.predictions enable row level security;

create policy "Users see own predictions"
  on public.predictions for all
  using (auth.uid() = user_id);

-- ── Audit Logs ─────────────────────────────────────────────
create table if not exists public.audit_logs (
  id          uuid primary key default gen_random_uuid(),
  user_id     uuid,
  action      text not null,
  resource    text,
  ip          text,
  detail      jsonb,
  timestamp   timestamptz default now()
);

alter table public.audit_logs enable row level security;

-- INSERT only — no UPDATE/DELETE
create policy "Audit logs insert only"
  on public.audit_logs for insert
  with check (true);

create policy "Admins read audit logs"
  on public.audit_logs for select
  using (
    exists (select 1 from public.users u where u.id = auth.uid() and u.role in ('admin','super_admin'))
  );

-- ── Realtime ───────────────────────────────────────────────
-- Enable Supabase Realtime on rounds table for live training charts
alter publication supabase_realtime add table public.rounds;
alter publication supabase_realtime add table public.experiments;

-- ── Sync Trigger ───────────────────────────────────────────
-- Sync auth.users to public.users on user creation
create or replace function public.handle_new_user()
returns trigger as $$
begin
  insert into public.users (id, email, role)
  values (
    new.id,
    new.email,
    coalesce(new.raw_app_meta_data->>'role', new.raw_user_meta_data->>'role', 'user')
  )
  on conflict (id) do update
  set email = excluded.email,
      role = coalesce(excluded.role, public.users.role);
  return new;
end;
$$ language plpgsql security definer;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute procedure public.handle_new_user();
