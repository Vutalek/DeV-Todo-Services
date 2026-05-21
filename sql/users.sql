create table if not exists users (
    id uuid primary key default gen_random_uuid(),
    login text not null,
    password text not null,
    created_at timestamp default now()
);

create index idx_users_login on users(login);