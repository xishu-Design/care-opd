import { lazy, Suspense } from "react";
import { createBrowserRouter, Navigate } from "react-router-dom";

import { App } from "../App";
import { AuthLayout } from "../layouts/AuthLayout";
import { DoctorLayout } from "../layouts/DoctorLayout";
import { PatientLayout } from "../layouts/PatientLayout";
import { StudentLayout } from "../layouts/StudentLayout";
import { LoginPage } from "../pages/common/LoginPage";
import { ChangePasswordPage } from "../pages/common/ChangePasswordPage";
import { NotFoundPage } from "../pages/common/NotFoundPage";
import { RegisterPage } from "../pages/common/RegisterPage";
import { RoleAIChatPage } from "../pages/common/RoleAIChatPage";
import { DoctorDashboardPage } from "../pages/doctor/DoctorDashboardPage";
import { DoctorHumanConfirmPage } from "../pages/doctor/DoctorHumanConfirmPage";
import { KnowledgeCatalogPage } from "../pages/doctor/KnowledgeCatalogPage";
import { KnowledgeSourcePage } from "../pages/doctor/KnowledgeSourcePage";
import { DoctorPatientDetailPage } from "../pages/doctor/DoctorPatientDetailPage";
import { DoctorPatientListPage } from "../pages/doctor/DoctorPatientListPage";
import { DoctorPrescriptionCreatePage } from "../pages/doctor/DoctorPrescriptionCreatePage";
import { DoctorPublishPage } from "../pages/doctor/DoctorPublishPage";
import { PatientDashboardPage } from "../pages/patient/PatientDashboardPage";
import { PatientEducationChatPage } from "../pages/patient/PatientEducationChatPage";
import { PatientPrescriptionDetailPage } from "../pages/patient/PatientPrescriptionDetailPage";
import { PatientPrescriptionListPage } from "../pages/patient/PatientPrescriptionListPage";
import { PatientProfilePage } from "../pages/patient/PatientProfilePage";
import { StudentCaseListPage } from "../pages/student/StudentCaseListPage";
import { StudentCaseTrainingPage } from "../pages/student/StudentCaseTrainingPage";
import { StudentDashboardPage } from "../pages/student/StudentDashboardPage";
import { StudentProgressPage } from "../pages/student/StudentProgressPage";
import { StudentQuizPage } from "../pages/student/StudentQuizPage";
import { ProtectedRoute } from "../routes/ProtectedRoute";
import { RedirectByRole } from "../routes/RedirectByRole";
import { RoleGuard } from "../routes/RoleGuard";

const DoctorKnowledgeGraphPage = lazy(() => import("../pages/doctor/DoctorKnowledgeGraphPage").then((module) => ({ default: module.DoctorKnowledgeGraphPage })));
const DoctorPrescriptionReviewPage = lazy(() => import("../pages/doctor/DoctorPrescriptionReviewPage").then((module) => ({ default: module.DoctorPrescriptionReviewPage })));
const StudentKnowledgeGraphPage = lazy(() => import("../pages/student/StudentKnowledgeGraphPage").then((module) => ({ default: module.StudentKnowledgeGraphPage })));
const StudentStandardAnalysisPage = lazy(() => import("../pages/student/StudentStandardAnalysisPage").then((module) => ({ default: module.StudentStandardAnalysisPage })));
const ContentQualityPage = lazy(() => import("../pages/harness/ContentQualityPage").then((module) => ({ default: module.ContentQualityPage })));
const AblationReportPage = lazy(() => import("../pages/harness/AblationReportPage").then((module) => ({ default: module.AblationReportPage })));
const ContentStatsPage = lazy(() => import("../pages/harness/ContentStatsPage").then((module) => ({ default: module.ContentStatsPage })));
const DataPipelinePage = lazy(() => import("../pages/harness/DataPipelinePage").then((module) => ({ default: module.DataPipelinePage })));
const DemoConsolePage = lazy(() => import("../pages/harness/DemoConsolePage").then((module) => ({ default: module.DemoConsolePage })));
const EvaluationDashboardPage = lazy(() => import("../pages/harness/EvaluationDashboardPage").then((module) => ({ default: module.EvaluationDashboardPage })));
const HarnessReportPage = lazy(() => import("../pages/harness/HarnessReportPage").then((module) => ({ default: module.HarnessReportPage })));
const LightRAGStatusPage = lazy(() => import("../pages/harness/LightRAGStatusPage").then((module) => ({ default: module.LightRAGStatusPage })));
const SourceRegistryPage = lazy(() => import("../pages/harness/SourceRegistryPage").then((module) => ({ default: module.SourceRegistryPage })));
const TraceDetailPage = lazy(() => import("../pages/harness/TraceDetailPage").then((module) => ({ default: module.TraceDetailPage })));

export const router = createBrowserRouter([
  {
    path: "/",
    element: <Suspense fallback={<div className="page-loading">页面加载中…</div>}><App /></Suspense>,
    children: [
      { index: true, element: <RedirectByRole /> },
      {
        element: <AuthLayout />,
        children: [
          { path: "role", element: <Navigate to="/login" replace /> },
          { path: "login", element: <LoginPage /> },
          { path: "register", element: <RegisterPage /> },
        ],
      },
      {
        element: <ProtectedRoute />,
        children: [
          { path: "account/password", element: <ChangePasswordPage /> },
          {
            element: <RoleGuard role="doctor" />,
            children: [
              {
                path: "doctor",
                element: <DoctorLayout />,
                children: [
                  { index: true, element: <Navigate to="/doctor/dashboard" replace /> },
                  { path: "dashboard", element: <DoctorDashboardPage /> },
                  { path: "ai-chat", element: <RoleAIChatPage role="doctor" /> },
                  { path: "patients", element: <DoctorPatientListPage /> },
                  { path: "patients/:patientId", element: <DoctorPatientDetailPage /> },
                  { path: "prescriptions/new", element: <DoctorPrescriptionCreatePage /> },
                  { path: "prescriptions/:prescriptionId/review", element: <DoctorPrescriptionReviewPage /> },
                  { path: "prescriptions/:prescriptionId/confirm", element: <DoctorHumanConfirmPage /> },
                  { path: "prescriptions/:prescriptionId/publish", element: <DoctorPublishPage /> },
                  { path: "knowledge-graph", element: <DoctorKnowledgeGraphPage /> },
                  { path: "knowledge/herbs", element: <KnowledgeCatalogPage entryType="herb" title="药材目录" subtitle="浏览中文药材名称、功效、关联证候与来源明确的传统配伍禁忌。" /> },
                  { path: "knowledge/formulas", element: <KnowledgeCatalogPage entryType="formula" title="方剂与组成关系" subtitle="展示数据集记录的方剂名称与组成关系，不构成处方建议。" /> },
                  { path: "knowledge/sources", element: <KnowledgeSourcePage /> },
                ],
              },
            ],
          },
          {
            element: <RoleGuard role="patient" />,
            children: [
              {
                path: "patient",
                element: <PatientLayout />,
                children: [
                  { index: true, element: <Navigate to="/patient/dashboard" replace /> },
                  { path: "dashboard", element: <PatientDashboardPage /> },
                  { path: "profile", element: <PatientProfilePage /> },
                  { path: "prescriptions", element: <PatientPrescriptionListPage /> },
                  { path: "prescriptions/:prescriptionId", element: <PatientPrescriptionDetailPage /> },
                  { path: "education", element: <PatientEducationChatPage /> },
                ],
              },
            ],
          },
          {
            element: <RoleGuard role="student" />,
            children: [
              {
                path: "student",
                element: <StudentLayout />,
                children: [
                  { index: true, element: <Navigate to="/student/dashboard" replace /> },
                  { path: "dashboard", element: <StudentDashboardPage /> },
                  { path: "ai-chat", element: <RoleAIChatPage role="student" /> },
                  { path: "cases", element: <StudentCaseListPage /> },
                  { path: "syndrome-cases", element: <StudentCaseListPage mode="syndrome" /> },
                  { path: "cases/:caseId/training", element: <StudentCaseTrainingPage /> },
                  { path: "cases/:caseId/analysis", element: <StudentStandardAnalysisPage /> },
                  { path: "quiz", element: <StudentQuizPage /> },
                  { path: "progress", element: <StudentProgressPage /> },
                  { path: "knowledge-graph", element: <StudentKnowledgeGraphPage /> },
                ],
              },
            ],
          },
          {
            element: <RoleGuard role="doctor" />,
            children: [
              {
                path: "harness",
                element: <DoctorLayout />,
                children: [
                  { index: true, element: <Navigate to="/harness/sources" replace /> },
                  { path: "demo", element: <DemoConsolePage /> },
                  { path: "evaluation", element: <EvaluationDashboardPage /> },
                  { path: "ablation", element: <AblationReportPage /> },
                  { path: "sources", element: <SourceRegistryPage /> },
                  { path: "data-pipeline", element: <DataPipelinePage /> },
                  { path: "content-stats", element: <ContentStatsPage /> },
                  { path: "content-quality", element: <ContentQualityPage /> },
                  { path: "lightrag", element: <LightRAGStatusPage /> },
                  { path: "traces", element: <TraceDetailPage /> },
                  { path: "traces/:traceId", element: <TraceDetailPage /> },
                  { path: "reports/:runId", element: <HarnessReportPage /> },
                ],
              },
            ],
          },
        ],
      },
      { path: "*", element: <NotFoundPage /> },
    ],
  },
]);
