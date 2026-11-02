import { Alert } from "antd";

import { errorMessage } from "../../utils/errors";

export function ErrorState({ error }: { error: unknown }) {
  return <Alert type="error" showIcon message={errorMessage(error)} />;
}
