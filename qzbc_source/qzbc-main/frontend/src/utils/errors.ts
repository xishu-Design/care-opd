import { ApiError } from "../api/http";

export function errorMessage(error: unknown) {
  if (error instanceof ApiError) {
    return error.errorCode ? `${error.message} (${error.errorCode})` : error.message;
  }
  if (error instanceof Error) return error.message;
  return "操作失败";
}
