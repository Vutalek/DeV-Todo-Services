create table if not exists permissions (
    role text not null,
    permission text not null
);

insert into permissions (role, permission)
values
('PROJECT_CREATOR', 'delete'),
('PROJECT_CREATOR', 'edit'),
('PROJECT_CREATOR', 'invite'),
('PROJECT_MEMBER', 'edit'),
('PROJECT_MEMBER', 'invite'),
('PROJECT_MEMBER', 'leave');