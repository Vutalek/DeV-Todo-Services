create table if not exists users_to_projects (
    user_id uuid not null,
    project_id uuid not null,
    role text not null

    constraint fk_users_to_projects_users
        foreign key (user_id) references users(id) 
        on delete cascade,
    
    constraint fk_users_to_projects_projects
        foreign key (project_id) references projects(id) 
        on delete cascade
);