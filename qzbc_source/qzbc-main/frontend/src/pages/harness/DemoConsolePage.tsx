import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Alert, App as AntApp, Button, Card, Descriptions, Space, Statistic, Table, Tag, Typography } from "antd";
import { ExternalLink, FlaskConical, Play, RefreshCw, RotateCcw } from "lucide-react";
import { useNavigate } from "react-router-dom";

import { demoApi } from "../../api/demoApi";
import { llmApi } from "../../api/llmApi";
import type { DemoSmokeCheck, LLMExperimentItem } from "../../api/types";
import { ErrorState } from "../../components/common/ErrorState";
import { LoadingState } from "../../components/common/LoadingState";
import { PageHeader } from "../../components/common/PageHeader";
import { StatusTag } from "../../components/common/StatusTag";
import { errorMessage } from "../../utils/errors";
import { asText, percent, shortId } from "../../utils/format";

export function DemoConsolePage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { modal, message } = AntApp.useApp();
  const status = useQuery({ queryKey: ["demo", "status"], queryFn: demoApi.getStatus });
  const reset = useMutation({
    mutationFn: () => demoApi.resetDemoData(),
    onSuccess: () => {
      message.success("演示数据已重置");
      queryClient.invalidateQueries({ queryKey: ["demo"] });
    },
  });
  const smoke = useMutation({
    mutationFn: demoApi.runSmoke,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["demo", "status"] }),
  });
  const experimentLatest = useQuery({
    queryKey: ["llm", "experiments", "latest"],
    queryFn: llmApi.getLatestExperiment,
    enabled: false,
    retry: false,
  });
  const experiment = useMutation({
    mutationFn: () => llmApi.runExperiment({ provider: "ollama", suite: "quick" }),
    onSuccess: (report) => {
      message.success("Ollama 快速实验已完成");
      queryClient.setQueryData(["llm", "experiments", "latest"], report);
      queryClient.invalidateQueries({ queryKey: ["demo", "status"] });
    },
  });
  const data = status.data;
  const resetDisabled = data?.environment.app_env !== "development";
  const experimentReport = experiment.data || experimentLatest.data;

  const confirmReset = () => {
    modal.confirm({
      title: "重置演示数据",
      content: "仅会校准固定 demo 账号与演示数据，development 环境可用。",
      okText: "重置",
      cancelText: "取消",
      onOk: () => reset.mutate(),
    });
  };

  return (
    <>
      <PageHeader
        title="演示控制台"
        subtitle="第 11 阶段演示验收与运行闭环。"
        extra={
          <Space wrap>
            <Button icon={<RefreshCw size={16} />} onClick={() => status.refetch()} loading={status.isFetching}>刷新</Button>
            <Button icon={<RotateCcw size={16} />} onClick={confirmReset} loading={reset.isPending} disabled={resetDisabled}>重置演示数据</Button>
            <Button type="primary" icon={<Play size={16} />} onClick={() => smoke.mutate()} loading={smoke.isPending}>冒烟检查</Button>
          </Space>
        }
      />

      <Space direction="vertical" size={16} style={{ width: "100%" }}>
        {reset.isError ? <Alert type="error" showIcon message={errorMessage(reset.error)} /> : null}
        {smoke.isError ? <Alert type="error" showIcon message={errorMessage(smoke.error)} /> : null}
        {experiment.isError ? <Alert type="error" showIcon message={errorMessage(experiment.error)} /> : null}
        {status.isLoading ? <LoadingState /> : status.isError ? <ErrorState error={status.error} /> : data ? (
          <>
            <div className="metric-grid">
              <Card><Statistic title="就绪状态" value={data.readiness.ready ? "就绪" : "受阻"} valueStyle={{ color: data.readiness.ready ? "#2f7d62" : "#c2410c" }} /></Card>
              <Card><Statistic title="阶段" value={data.environment.phase} /></Card>
              <Card><Statistic title="版本" value={data.environment.version} /></Card>
              <Card><Statistic title="应用环境" value={data.environment.app_env} /></Card>
            </div>

            <div className="metric-grid">
              <Card><Statistic title="知识条目" value={data.content.knowledge_entries || 0} /></Card>
              <Card><Statistic title="规则" value={data.content.rules || 0} /></Card>
              <Card><Statistic title="学习案例" value={data.content.learning_cases || 0} /></Card>
              <Card><Statistic title="测验" value={data.content.quizzes || 0} /></Card>
            </div>

            <Card title="就绪检查">
              <Space direction="vertical" size={8} style={{ width: "100%" }}>
                <Space wrap>
                  <Tag color={data.backend.database_ok ? "green" : "red"}>数据库</Tag>
                  <Tag color={data.backend.alembic_current ? "green" : "red"}>数据库迁移</Tag>
                  <Tag color={data.llm.fallback_available ? "green" : "gold"}>LLM 降级</Tag>
                  <Tag color={data.lightrag.enabled ? "green" : "gold"}>lightrag {data.lightrag.status}</Tag>
                </Space>
                {data.readiness.blocking_issues.length ? (
                  <Alert type="error" showIcon message={data.readiness.blocking_issues.join(" / ")} />
                ) : null}
                {data.readiness.warnings.length ? (
                  <Alert type="warning" showIcon message={data.readiness.warnings.join(" / ")} />
                ) : null}
              </Space>
            </Card>

            <div className="two-column">
              <Card title="演示账号">
                <Descriptions column={1} bordered size="small">
                  <Descriptions.Item label="doctor_demo"><StatusTag value={data.accounts.doctor_demo_exists ? "success" : "failed"} /></Descriptions.Item>
                  <Descriptions.Item label="patient_demo"><StatusTag value={data.accounts.patient_demo_exists ? "success" : "failed"} /></Descriptions.Item>
                  <Descriptions.Item label="student_demo"><StatusTag value={data.accounts.student_demo_exists ? "success" : "failed"} /></Descriptions.Item>
                  <Descriptions.Item label="医患绑定"><StatusTag value={data.bindings.doctor_patient_binding_exists ? "success" : "failed"} /></Descriptions.Item>
                </Descriptions>
              </Card>

              <Card title="系统能力">
                <Descriptions column={1} bordered size="small">
                  <Descriptions.Item label="LLM">{data.llm.provider} / {data.llm.status} / {asText(data.llm.model)}</Descriptions.Item>
                  <Descriptions.Item label="LightRAG">{data.lightrag.backend} / {data.lightrag.status} / docs {data.lightrag.docs_count || 0}</Descriptions.Item>
                  <Descriptions.Item label="Harness">{data.harness.last_run_id ? `${data.harness.last_phase} (${percent(Number(data.harness.last_pass_rate))})` : "not_run"}</Descriptions.Item>
                  <Descriptions.Item label="演示重置">{data.demo_reset?.last_reset_id ? `${shortId(data.demo_reset.last_reset_id)} / ${data.demo_reset.last_status}` : "未运行"}</Descriptions.Item>
                  <Descriptions.Item label="链路追踪">{shortId(data.observability?.latest_trace_id)}</Descriptions.Item>
                  <Descriptions.Item label="证据包">{shortId(data.observability?.latest_evidence_packet_id)}</Descriptions.Item>
                </Descriptions>
              </Card>
            </div>

            <Card title="LLM 健康状态">
              <Space direction="vertical" size={12} style={{ width: "100%" }}>
                <Descriptions column={2} bordered size="small">
                  <Descriptions.Item label="active">{data.llm.active_provider || data.llm.provider}</Descriptions.Item>
                  <Descriptions.Item label="configured">{data.llm.configured_provider || data.llm.provider}</Descriptions.Item>
                  <Descriptions.Item label="status"><Tag>{data.llm.status}</Tag></Descriptions.Item>
                  <Descriptions.Item label="reachable"><Tag color={data.llm.reachable ? "green" : "gold"}>{String(Boolean(data.llm.reachable))}</Tag></Descriptions.Item>
                  <Descriptions.Item label="fallback">{data.llm.fallback_provider || "template"} / {String(data.llm.fallback_available)}</Descriptions.Item>
                  <Descriptions.Item label="model">{asText(data.llm.model || "-")}</Descriptions.Item>
                  <Descriptions.Item label="latency">{data.llm.latency_ms ?? "-"} ms</Descriptions.Item>
                  <Descriptions.Item label="last_error">{data.llm.last_error ? <Typography.Text type="danger">{data.llm.last_error}</Typography.Text> : "-"}</Descriptions.Item>
                </Descriptions>
                {data.llm.status !== "ok" ? (
                  <Alert type="warning" showIcon message="LLM 主提供方不可用或已降级；模板回退仍可保证基础演示。" />
                ) : null}
                <Table
                  size="small"
                  rowKey="name"
                  pagination={false}
                  dataSource={data.llm.providers || []}
                  columns={[
                    { title: "provider", dataIndex: "name" },
                    { title: "configured", dataIndex: "configured", render: (value) => <Tag color={value ? "green" : "gold"}>{String(Boolean(value))}</Tag> },
                    { title: "reachable", dataIndex: "reachable", render: (value) => <Tag color={value ? "green" : "gold"}>{String(Boolean(value))}</Tag> },
                    { title: "status", dataIndex: "status" },
                    { title: "model", dataIndex: "model", render: (value) => asText(value || "-") },
                    { title: "latency", dataIndex: "latency_ms", render: (value) => (value == null ? "-" : `${value} ms`) },
                  ]}
                />
              </Space>
            </Card>

            <Card
              title="LLM 实验"
              extra={
                <Space wrap>
                  <Button icon={<RefreshCw size={16} />} onClick={() => experimentLatest.refetch()} loading={experimentLatest.isFetching}>刷新最近报告</Button>
                  <Button type="primary" icon={<FlaskConical size={16} />} onClick={() => experiment.mutate()} loading={experiment.isPending}>Ollama 快速实验</Button>
                </Space>
              }
            >
              <Space direction="vertical" size={12} style={{ width: "100%" }}>
                {experimentReport ? (
                  <>
                    {experimentReport.fallback_count > 0 ? (
                      <Alert type="warning" showIcon message="一个或多个实验用例使用了模板回退。" />
                    ) : null}
                    <div className="metric-grid">
                      <Statistic title="提供方" value={experimentReport.provider} />
                      <Statistic title="用例总数" value={experimentReport.total_cases} />
                      <Statistic title="回退次数" value={experimentReport.fallback_count} />
                      <Statistic title="平均延迟" value={experimentReport.avg_latency_ms} suffix="ms" />
                    </div>
                    <div className="metric-grid">
                      <Statistic title="解析成功率" value={percent(experimentReport.parse_success_rate)} />
                      <Statistic title="结构校验率" value={percent(experimentReport.schema_valid_rate)} />
                      <Statistic title="安全通过率" value={percent(experimentReport.safety_pass_rate)} />
                      <Statistic title="通过数" value={`${experimentReport.success}/${experimentReport.total_cases}`} />
                    </div>
                    <Table<LLMExperimentItem>
                      size="small"
                      rowKey="case_id"
                      dataSource={experimentReport.items}
                      pagination={false}
                      columns={[
                        { title: "case", dataIndex: "case_id" },
                        { title: "task", dataIndex: "task_type" },
                        { title: "结果", render: (_, record) => <StatusTag value={record.passed ? "success" : "failed"} /> },
                        { title: "provider", dataIndex: "provider" },
                        { title: "fallback", dataIndex: "fallback_used", render: (value) => <Tag color={value ? "gold" : "green"}>{String(Boolean(value))}</Tag> },
                        { title: "schema", dataIndex: "schema_valid", render: (value) => <Tag color={value ? "green" : "red"}>{String(Boolean(value))}</Tag> },
                      ]}
                    />
                  </>
                ) : (
                  <Typography.Text type="secondary">暂无 LLM 实验报告。</Typography.Text>
                )}
              </Space>
            </Card>

            <Card title="演示数据">
              <Descriptions column={2} bordered size="small">
                {Object.entries(data.demo_data).map(([key, value]) => (
                  <Descriptions.Item key={key} label={key}>
                    <StatusTag value={value ? "success" : "failed"} />
                  </Descriptions.Item>
                ))}
              </Descriptions>
            </Card>

            <Card title="快捷操作">
              <Space wrap>
                <Button icon={<ExternalLink size={16} />} onClick={() => navigate("/doctor/patients")}>医生患者</Button>
                <Button icon={<ExternalLink size={16} />} onClick={() => navigate("/doctor/prescriptions/new")}>医生拟方</Button>
                <Button icon={<ExternalLink size={16} />} onClick={() => navigate("/harness/lightrag")}>LightRAG</Button>
                <Button icon={<ExternalLink size={16} />} onClick={() => navigate("/harness/traces")}>链路追踪</Button>
                <Button icon={<ExternalLink size={16} />} onClick={() => navigate(data.harness.last_run_id ? `/harness/reports/${data.harness.last_run_id}` : "/harness/reports/0")}>验收报告</Button>
              </Space>
            </Card>
          </>
        ) : (
          <Card><Typography.Text type="secondary">暂无演示状态。</Typography.Text></Card>
        )}

        {smoke.data ? (
          <Card title="冒烟检查">
            <div className="metric-grid" style={{ marginBottom: 16 }}>
              <Statistic title="结果" value={smoke.data.passed ? "通过" : "失败"} valueStyle={{ color: smoke.data.passed ? "#2f7d62" : "#c2410c" }} />
              <Statistic title="检查总数" value={smoke.data.total_checks} />
              <Statistic title="通过" value={smoke.data.passed_checks} />
              <Statistic title="失败" value={smoke.data.failed_checks} />
            </div>
            {smoke.data.warnings.length ? <Alert type="warning" showIcon message={smoke.data.warnings.join(" / ")} style={{ marginBottom: 16 }} /> : null}
            <Table<DemoSmokeCheck>
              rowKey="name"
              dataSource={smoke.data.checks}
              pagination={false}
              columns={[
                { title: "检查项", dataIndex: "name" },
                { title: "结果", render: (_, record) => <StatusTag value={record.passed ? "success" : "failed"} /> },
                { title: "严重程度", dataIndex: "severity", render: (value) => <Tag>{value}</Tag> },
                { title: "详情", dataIndex: "details" },
              ]}
            />
          </Card>
        ) : null}
      </Space>
    </>
  );
}
