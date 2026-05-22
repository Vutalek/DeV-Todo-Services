import os

from pwdlib import PasswordHash
import sqlalchemy as sa
import jwt
from dotenv import load_dotenv
load_dotenv()

from db_handler import DBHandler


class AuthHandler:
    def __init__(self):
        self.__secret_key = os.environ.get("JWT_SECRET_KEY", "")
        self.__algorithm = os.environ.get("JWT_ALGORITHM", "")

        self.password_hasher = PasswordHash.recommended()
        self.dummy_hash = self.password_hasher.hash("dummy_password")
        self.db = DBHandler()

    def hash_password(self, password: str) -> str:
        return self.password_hasher.hash(password)

    def verify_password(self, password: str, hashed_password: str) -> bool:
        return self.password_hasher.verify(password, hashed_password)
    
    def authenticate_user(self, username: str, password: str) -> bool:
        with self.db.engine.connect() as connection:
            result = connection.execute(
                sa.text("select * from users where username = :username"),
                {"username": username}
            )
            user = result.fetchone()
        if not user:
            return self.verify_password(password, self.dummy_hash)
        return self.verify_password(password, user.password_hash)

    def create_access_token(self, data: dict) -> str:
        to_encode = data.copy()
        encoded_jwt = jwt.encode(to_encode, self.__secret_key, algorithm=self.__algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> str | None:
        try:
            payload = jwt.decode(token, os.environ.get("JWT_SECRET_KEY"), algorithms=[os.environ.get("JWT_ALGORITHM", "")])
            username = payload.get("sub")
            if username is None:
                raise jwt.InvalidTokenError
            return username
        except jwt.InvalidTokenError:
            return None