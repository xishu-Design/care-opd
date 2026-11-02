from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.enums import UserRole
from app.core.security import create_access_token, hash_password, verify_password
from app.db.models import DoctorProfile, PatientProfile, StudentProfile, User
from app.schemas.auth import ChangePasswordRequest, LoginRequest, RegisterRequest


def get_user_by_username(db: Session, username: str) -> User | None:
    return db.scalar(select(User).where(User.username == username))


def create_role_profile(db: Session, user: User) -> None:
    if user.role == UserRole.DOCTOR.value:
        db.add(DoctorProfile(user_id=user.id))
    elif user.role == UserRole.PATIENT.value:
        db.add(PatientProfile(user_id=user.id))
    elif user.role == UserRole.STUDENT.value:
        db.add(StudentProfile(user_id=user.id))


def register_user(db: Session, payload: RegisterRequest) -> User:
    if payload.role == UserRole.ADMIN:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Admin registration is not enabled")
    if get_user_by_username(db, payload.username):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username already exists")

    user = User(
        username=payload.username,
        hashed_password=hash_password(payload.password),
        role=payload.role.value,
        display_name=payload.display_name,
    )
    db.add(user)
    db.flush()
    create_role_profile(db, user)
    db.commit()
    db.refresh(user)
    return user


def authenticate_user(db: Session, payload: LoginRequest) -> tuple[str, User]:
    user = get_user_by_username(db, payload.username)
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is disabled")
    if user.role != payload.entry_role.value:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Selected entry role does not match account role")

    token = create_access_token(subject=str(user.id), claims={"role": user.role})
    return token, user


def change_password(db: Session, user: User, payload: ChangePasswordRequest) -> User:
    if not verify_password(payload.current_password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="当前密码不正确")
    if payload.new_password != payload.confirm_password:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="两次输入的新密码不一致")
    if verify_password(payload.new_password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="新密码不能与当前密码相同")
    user.hashed_password = hash_password(payload.new_password)
    user.must_change_password = False
    db.commit()
    db.refresh(user)
    return user
