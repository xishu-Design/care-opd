from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db
from app.core.responses import success_response
from app.schemas.auth import ChangePasswordRequest, LoginRequest, RegisterRequest, TokenResponse
from app.schemas.user import UserRead
from app.services.auth_service import authenticate_user, change_password, register_user

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register")
async def register(payload: RegisterRequest, db: Session = Depends(get_db)) -> dict:
    user = register_user(db, payload)
    return success_response(UserRead.model_validate(user).model_dump(mode="json"), "registered")


@router.post("/login")
async def login(payload: LoginRequest, db: Session = Depends(get_db)) -> dict:
    token, user = authenticate_user(db, payload)
    data = TokenResponse(
        access_token=token,
        user={
            "id": user.id,
            "username": user.username,
            "role": user.role,
            "display_name": user.display_name,
            "must_change_password": user.must_change_password,
        },
    )
    return success_response(data.model_dump(), "logged in")


@router.get("/me")
async def me(current_user=Depends(get_current_user)) -> dict:
    return success_response(UserRead.model_validate(current_user).model_dump(mode="json"))


@router.post("/change-password")
async def update_password(
    payload: ChangePasswordRequest,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    user = change_password(db, current_user, payload)
    return success_response(UserRead.model_validate(user).model_dump(mode="json"), "密码修改成功")
