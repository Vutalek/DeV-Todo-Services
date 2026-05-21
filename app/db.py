import os
from typing import List, Tuple
import logging
logging.basicConfig(format="%(name)s: %(asctime)s - [%(levelname)s]: %(message)s", level=logging.NOTSET)
logger = logging.getLogger("db")

import sqlalchemy as sa
from pwdlib import PasswordHash
from dotenv import load_dotenv
load_dotenv()


class DBFacade:
    def __init__(self):
        self.__host = os.environ.get("POSTGRES_HOST", "")
        self.__db = os.environ.get("POSTGRES_DB", "")
        self.__user = os.environ.get("POSTGRES_USER", "")
        self.__password = os.environ.get("POSTGRES_PASSWORD", "")
        self.engine = sa.create_engine(f"postgresql+asynpg://{self.__user}:{self.__password}@{self.__host}/{self.__db}")
        self.hasher = PasswordHash.recommended()
    
    ### USER LOGIC

    def create_user(self, login: str, password: str) -> bool:
        try:
            pwd_hash = self.hasher.hash(password)
            with self.engine.connect() as conn:
                conn.execute(
                    sa.text("insert into users (login, password) values (:login, :password)"),
                    {"login": login, "password": pwd_hash}
                )
                conn.commit()
            logger.debug(f"Create new user {login}.")
            return True
        except Exception as e:
            logger.error(f"Failed creating new user {login}: {e}.")
            return False
        
    def get_user_password_hash(self, login: str) -> str:
        with self.engine.connect() as conn:
            result = conn.execute(
                sa.text("select password from users where login=:login"),
                {"login": login}
            ).fetchone()
        if result is not None:
            return result.password
        else:
            return ""
        
    def get_user_id(self, login: str) -> str:
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    sa.text("select id from users where login = :login"),
                    {"login": login}
                ).fetchone()
                if result is None:
                    logger.error(f"Failed to get user {login}.")
                    return ""
            return result.id
        except Exception as e:
            logger.error(f"Failed to get user {login}: {e}.")
            return ""
    
    ### PROJECT LOGIC
        
    def create_project(self, login: str, name: str, description: str) -> bool:
        user = self.get_user_id(login)
        if user == "":
            logger.error(f"Failed to get user {login}/{name} from DB.")
            return False
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    sa.text("insert into projects (name, description) values (:name, :description) returning id"),
                    {"name": name, "description": description}
                ).fetchone()
                if result is None:
                    logger.error(f"Failed creating new project {login}/{name} in DB.")
                    return False
                
                conn.execute(
                    sa.text("insert into users_to_projects (user_id, project_id, role) values (:user, :project, :role)"),
                    {"user": user, "project": result.id, "role": "PROJECT_CREATOR"}
                )
                conn.commit()
            logger.debug(f"Created new project {login}/{name}.")
            return True
        except Exception as e:
            logger.error(f"Failed creating new project {login}/{name}: {e}.")
            return False
    
    def get_user_projects(self, login: str) -> List[Tuple[str, str, str]]:
        user_id = self.get_user_id(login)
        if user_id == "":
            return []
        
        try:
            with self.engine.connect() as conn:                
                projects = conn.execute(
                    sa.text(
                        "select utp.project_id, p.name, p.description" \
                        "from users_to_projects as utp" \
                        "join projects as p" \
                        "by utp.project_id = p.id" \
                        "where utp.user_id = :user_id"
                    ),
                    {"user_id": user_id}
                ).fetchall()
            projects = [p._tuple() for p in projects]
            return projects
        except Exception as e:
            logger.error(f"Failed to fetch {login} projects: {e}.")
            return []
        
    def get_project_members(self, login: str, project: str) -> List[Tuple[str, str, str]]:
        project_id = self.get_project_id(login, project)
        if project_id == "":
            return []
        
        try:
            with self.engine.connect() as conn:                
                users = conn.execute(
                    sa.text(
                        "select utp.user_id, u.login, utp.role" \
                        "from users_to_projects as utp" \
                        "join users as u" \
                        "by utp.user_id = u.id" \
                        "where utp.project_id = :project_id"
                    ),
                    {"project_id": project_id}
                ).fetchall()
            users = [u._tuple() for u in users]
            return users
        except Exception as e:
            logger.error(f"Failed to fetch {project} users: {e}.")
            return []
        
    def get_project_id(self, login: str, project: str) -> str:
        user_projects = self.get_user_projects(login)
        result = ""
        for proj in user_projects:
            if proj[1] == project:
                result = proj[0]
                break
        return result
        
    def get_user_permissions(self, login: str, project: str) -> List[str]:
        user_id = self.get_user_id(login)
        if user_id == "":
            logger.error(f"Failed to get user {login} from DB.")
            return []
        
        project_id = self.get_project_id(login, project)
        if project_id == "":
            logger.error(f"Failed to get project {login}/{project} from DB.")
            return []
        
        try:
            with self.engine.connect() as conn:
                role = conn.execute(
                    sa.text(
                        "select role" \
                        "from users_to_projects"
                        "where user_id = :user_id and project_id = :project_id"
                    ),
                    {"user_id": user_id, "project_id": project_id}
                ).fetchone()
                if role is None:
                    return []
                
                result = conn.execute(
                    sa.text("select permission from permissions where role = :role"),
                    {"role": role.role}
                ).fetchall()
                if result is None:
                    logger.error(f"Tryed to get permissions from non existing role.")
                    return []
            return [p.permission for p in result]
        except Exception as e:
            logger.error(f"Failed to get permissions {login} for {project}.")
            return []
        
    def add_member_to_project(self, login_from: str, login_to: str, project: str) -> bool:
        permissions = self.get_user_permissions(login_from, project)
        if "invite" not in permissions:
            return False

        project_id = self.get_project_id(login_from, project)
        if project_id == "":
            return False
        
        user_id = self.get_user_id(login_to)
        if user_id == "":
            return False

        try:
            with self.engine.connect() as conn:
                conn.execute(
                    sa.text("insert into users_to_projects (user_id, project_id, role) values (:user, :project, :role)"),
                    {"user": user_id, "project": project_id, "role": "PROJECT_MEMBER"}
                )
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to add member {login_to} for {project}.")
            return False
        