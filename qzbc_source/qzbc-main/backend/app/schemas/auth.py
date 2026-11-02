from pydantic import BaseModel, Field

from app.core.enums import UserRole


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=80)
    password: str = Field(min_length=6, max_length=128)
    role: UserRole
    display_name: str = Field(min_length=1, max_length=120)


class LoginRequest(BaseModel):
    username: str
    password: str
    entry_role: UserRole


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=6, max_length=128)
    new_password: str = Field(min_length=8, max_length=128)
    confirm_password: str = Field(min_length=8, max_length=128)


class TokenUser(BaseModel):
    id: int
    username: str
    role: str
    display_name: str
    must_change_password: bool = False


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: TokenUser
