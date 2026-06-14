create table if not exists projects (
    id uuid primary key default gen_random_uuid(),
    name text not null,
    description text not null,
    created_at timestamp default now()
);