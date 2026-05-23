import os

from pwdlib import PasswordHash
import jwt
from dotenv import load_dotenv
load_dotenv()

from .db import DBFacade


class AuthHandler:
    def __init__(self):
        self.__secret_key = os.environ.get("JWT_SECRET_KEY", "")
        self.__algorithm = os.environ.get("JWT_ALGORITHM", "")

        self.password_hasher = PasswordHash.recommended()
        self.dummy_hash = self.password_hasher.hash("dummy_password")
        self.db = DBFacade()

    def hash_password(self, password: str) -> str:
        return self.password_hasher.hash(password)

    def verify_password(self, password: str, hashed_password: str) -> bool:
        return self.password_hasher.verify(password, hashed_password)
    
    async def authenticate_user(self, login: str, password: str) -> bool:
        pwd_hash = await self.db.get_user_password_hash(login)
        if not pwd_hash:
            return self.verify_password(password, self.dummy_hash)
        return self.verify_password(password, pwd_hash)

    def create_access_token(self, data: dict) -> str:
        to_encode = data.copy()
        encoded_jwt = jwt.encode(to_encode, self.__secret_key, algorithm=self.__algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> str | None:
        try:
            payload = jwt.decode(token, os.environ.get("JWT_SECRET_KEY", ""), algorithms=[os.environ.get("JWT_ALGORITHM", "")])
            login = payload.get("sub")
            if login is None:
                raise jwt.InvalidTokenError
            return login
        except jwt.InvalidTokenError:
            return None