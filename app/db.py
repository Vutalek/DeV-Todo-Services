import os
from typing import Any, List, Tuple
import logging
import asyncio

import asyncpg
from pwdlib import PasswordHash
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    format="%(name)s: %(asctime)s - [%(levelname)s]: %(message)s",
    level=logging.NOTSET,
)
logger = logging.getLogger("db")


class DBFacade:
    def __init__(self):
        self.__host = os.environ.get("POSTGRES_HOST", "")
        self.__port = int(os.environ.get("POSTGRES_PORT", ""))
        self.__db = os.environ.get("POSTGRES_DB", "")
        self.__user = os.environ.get("POSTGRES_USER", "")
        self.__password = os.environ.get("POSTGRES_PASSWORD", "")
        self._pool: asyncpg.Pool | None = None
        self._pool_lock = asyncio.Lock()
        self.hasher = PasswordHash.recommended()

    async def connect(self) -> asyncpg.Pool:
        if self._pool is None:
            async with self._pool_lock:
                if self._pool is None:
                    self._pool = await asyncpg.create_pool(
                        host=self.__host,
                        port=self.__port,
                        database=self.__db,
                        user=self.__user,
                        password=self.__password,
                    )
        return self._pool

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def _get_user_id(self, login: str, conn: asyncpg.Connection) -> Any:
        result = await conn.fetchrow(
            "select id from users where login = $1",
            login,
        )
        if result is None:
            return ""
        return result["id"]

    async def _get_project_id(
        self,
        login: str,
        project: str,
        conn: asyncpg.Connection,
    ) -> Any:
        result = await conn.fetchrow(
            "select p.id "
            "from users_to_projects as utp "
            "join projects as p "
            "on utp.project_id = p.id "
            "join users as u "
            "on utp.user_id = u.id "
            "where u.login = $1 and p.name = $2",
            login,
            project,
        )
        if result is None:
            return ""
        return result["id"]

    ### USER LOGIC

    async def create_user(self, login: str, password: str) -> bool:
        try:
            pwd_hash = self.hasher.hash(password)
            pool = await self.connect()
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute(
                        "insert into users (login, password) values ($1, $2)",
                        login,
                        pwd_hash,
                    )
            logger.debug(f"Create new user {login}.")
            return True
        except Exception as e:
            logger.error(f"Failed creating new user {login}: {e}.")
            return False

    async def get_user_password_hash(self, login: str) -> str:
        pool = await self.connect()
        async with pool.acquire() as conn:
            result = await conn.fetchrow(
                "select password from users where login = $1",
                login,
            )
        if result is not None:
            return result["password"]
        return ""

    async def get_user_id(self, login: str) -> Any:
        try:
            pool = await self.connect()
            async with pool.acquire() as conn:
                return await self._get_user_id(login, conn)
        except Exception as e:
            logger.error(f"Failed to get user {login}: {e}.")
            return ""

    ### PROJECT LOGIC

    async def create_project(self, login: str, name: str, description: str) -> bool:
        try:
            pool = await self.connect()
            async with pool.acquire() as conn:
                async with conn.transaction():
                    user = await self._get_user_id(login, conn)
                    if user == "":
                        logger.error(f"Failed to get user {login}/{name} from DB.")
                        return False

                    result = await conn.fetchrow(
                        "insert into projects (name, description) "
                        "values ($1, $2) "
                        "returning id",
                        name,
                        description,
                    )
                    if result is None:
                        raise RuntimeError(f"Failed creating new project {login}/{name} in DB.")

                    await conn.execute(
                        "insert into users_to_projects (user_id, project_id, role) "
                        "values ($1, $2, $3)",
                        user,
                        result["id"],
                        "PROJECT_CREATOR",
                    )
            logger.debug(f"Created new project {login}/{name}.")
            return True
        except Exception as e:
            logger.error(f"Failed creating new project {login}/{name}: {e}.")
            return False

    async def get_user_projects(self, login: str) -> List[Tuple[Any, str, str]]:
        try:
            pool = await self.connect()
            async with pool.acquire() as conn:
                user_id = await self._get_user_id(login, conn)
                if user_id == "":
                    return []

                projects = await conn.fetch(
                    "select utp.project_id, p.name, p.description "
                    "from users_to_projects as utp "
                    "join projects as p "
                    "on utp.project_id = p.id "
                    "where utp.user_id = $1",
                    user_id,
                )
            return [tuple(project) for project in projects]
        except Exception as e:
            logger.error(f"Failed to fetch {login} projects: {e}.")
            return []

    async def get_project_members(self, login: str, project: str) -> List[Tuple[Any, str, str]]:
        try:
            pool = await self.connect()
            async with pool.acquire() as conn:
                project_id = await self._get_project_id(login, project, conn)
                if project_id == "":
                    return []

                users = await conn.fetch(
                    "select utp.user_id, u.login, utp.role "
                    "from users_to_projects as utp "
                    "join users as u "
                    "on utp.user_id = u.id "
                    "where utp.project_id = $1",
                    project_id,
                )
            return [tuple(user) for user in users]
        except Exception as e:
            logger.error(f"Failed to fetch {project} users: {e}.")
            return []

    async def get_project_id(self, login: str, project: str) -> Any:
        try:
            pool = await self.connect()
            async with pool.acquire() as conn:
                return await self._get_project_id(login, project, conn)
        except Exception as e:
            logger.error(f"Failed to get project {login}/{project}: {e}.")
            return ""

    async def get_user_permissions(self, login: str, project: str) -> List[str]:
        try:
            pool = await self.connect()
            async with pool.acquire() as conn:
                user_id = await self._get_user_id(login, conn)
                if user_id == "":
                    logger.error(f"Failed to get user {login} from DB.")
                    return []

                project_id = await self._get_project_id(login, project, conn)
                if project_id == "":
                    logger.error(f"Failed to get project {login}/{project} from DB.")
                    return []

                role = await conn.fetchrow(
                    "select role "
                    "from users_to_projects "
                    "where user_id = $1 and project_id = $2",
                    user_id,
                    project_id,
                )
                if role is None:
                    return []

                result = await conn.fetch(
                    "select permission from permissions where role = $1",
                    role["role"],
                )
                if result is None:
                    logger.error("Tried to get permissions from non existing role.")
                    return []
            return [permission["permission"] for permission in result]
        except Exception as e:
            logger.error(f"Failed to get permissions {login} for {project}: {e}.")
            return []

    async def add_member_to_project(
        self,
        login_from: str,
        login_to: str,
        project: str,
    ) -> bool:
        permissions = await self.get_user_permissions(login_from, project)
        if "invite" not in permissions:
            return False

        try:
            pool = await self.connect()
            async with pool.acquire() as conn:
                async with conn.transaction():
                    project_id = await self._get_project_id(login_from, project, conn)
                    if project_id == "":
                        return False

                    user_id = await self._get_user_id(login_to, conn)
                    if user_id == "":
                        return False

                    await conn.execute(
                        "insert into users_to_projects (user_id, project_id, role) "
                        "values ($1, $2, $3)",
                        user_id,
                        project_id,
                        "PROJECT_MEMBER",
                    )
            return True
        except Exception as e:
            logger.error(f"Failed to add member {login_to} for {project}: {e}.")
            return False
        
    async def leave_project(
        self,
        login: str,
        project: str,
    ) -> bool:
        permissions = await self.get_user_permissions(login, project)
        if "leave" not in permissions:
            return False

        try:
            pool = await self.connect()
            async with pool.acquire() as conn:
                async with conn.transaction():
                    project_id = await self._get_project_id(login, project, conn)
                    if project_id == "":
                        return False

                    user_id = await self._get_user_id(login, conn)
                    if user_id == "":
                        return False

                    await conn.execute(
                        "delete from users_to_projects "
                        "where project_id = $1 and user_id = $2",
                        user_id,
                        project_id,
                    )
            return True
        except Exception as e:
            logger.error(f"Failed {login} to leave {project}: {e}.")
            return False

    async def delete_project(self, login: str, project: str) -> bool:
        permissions = await self.get_user_permissions(login, project)
        if "delete" not in permissions:
            return False

        try:
            pool = await self.connect()
            async with pool.acquire() as conn:
                async with conn.transaction():
                    project_id = await self._get_project_id(login, project, conn)
                    if project_id == "":
                        return False

                    await conn.execute(
                        "delete from projects where id = $1",
                        project_id,
                    )
            return True
        except Exception as e:
            logger.error(f"Failed to delete project {login}/{project}: {e}.")
            return False
