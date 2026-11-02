from app.content.services.content_import_service import ContentImportService
from app.content.services.content_quality_service import content_quality_report
from app.tests.test_phase9_content_import import _content_fixture


def test_content_quality_report_has_source_coverage_and_no_issues(db_session, tmp_path):
    ContentImportService(db_session, _content_fixture(tmp_path)).import_content()

    report = content_quality_report(db_session)

    assert report["metrics"]["content_source_coverage_rate"] == 1.0
    assert report["metrics"]["learning_case_deid_pass_rate"] == 1.0
    assert report["critical_issues"] == []
    assert report["high_issues"] == []

