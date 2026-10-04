from __future__ import annotations

import asyncio
import hashlib
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from mcp import Client
from starlette.testclient import TestClient

from mcp_readonly import server as readonly


ROOT = Path(__file__).resolve().parents[2]


def _hash_whitelist() -> dict[str, str]:
    catalog = readonly._load_catalog()
    result: dict[str, str] = {}
    for entry in catalog["documents"]:
        path = ROOT / entry["path"]
        result[entry["path"]] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


class ReadOnlyMCPTests(unittest.TestCase):
    def test_current_search_follows_authority_without_overriding_version_query(self) -> None:
        current = readonly.search("当前模型能力和下一步方向")
        self.assertEqual(current.results[0].id, "v169-platform-package-ready")
        historical = readonly.search("v7.7 同输入失败归因")
        self.assertEqual(historical.results[0].id, "v77-direction-review")

    def test_catalog_documents_exist_and_match_hashes(self) -> None:
        catalog = readonly._load_catalog()
        for entry in readonly._documents_by_id(catalog).values():
            path, content = readonly._safe_document(entry)
            self.assertTrue(path.is_file())
            self.assertLessEqual(len(content), readonly.MAX_DOCUMENT_BYTES)

    def test_tools_are_only_expected_read_only_operations(self) -> None:
        async def check() -> None:
            async with Client(readonly.server) as client:
                result = await client.list_tools()
                tools = {tool.name: tool for tool in result.tools}
                self.assertEqual(set(tools), {"search", "fetch", "get_project_status"})
                for tool in tools.values():
                    self.assertTrue(tool.annotations.read_only_hint)
                    self.assertFalse(tool.annotations.destructive_hint)
                    self.assertTrue(tool.annotations.idempotent_hint)
                    self.assertFalse(tool.annotations.open_world_hint)
                    properties = tool.input_schema.get("properties", {})
                    self.assertFalse({"path", "command", "url", "content"} & set(properties))

        asyncio.run(check())

    def test_search_fetch_and_status_leave_sources_unchanged(self) -> None:
        before = _hash_whitelist()

        async def check() -> None:
            async with Client(readonly.server) as client:
                search_result = await client.call_tool("search", {"query": "当前模型能力和下一步方向"})
                self.assertFalse(search_result.is_error)
                self.assertIn("v169-platform-package-ready", json.dumps(search_result.structured_content, ensure_ascii=False))

                fetch_result = await client.call_tool("fetch", {"id": "v169-platform-package-ready"})
                self.assertFalse(fetch_result.is_error)
                self.assertIn("V159", json.dumps(fetch_result.structured_content, ensure_ascii=False))

                status_result = await client.call_tool("get_project_status", {})
                self.assertFalse(status_result.is_error)
                status = status_result.structured_content
                self.assertFalse(status["quality_acceptance"])
                catalog=readonly._load_catalog()
                expected=json.loads((ROOT/readonly._documents_by_id(catalog)[catalog["project"]["authoritative_delivery_id"]]["path"]).read_text(encoding="utf-8"))
                self.assertEqual(status["experiment_status"],expected["status"])
                self.assertEqual(set(status["source_ids"]), {catalog["project"]["authoritative_delivery_id"], "v169-platform-package-ready"})
                delivery = await client.call_tool("fetch", {"id": catalog["project"]["authoritative_delivery_id"]})
                self.assertFalse(delivery.is_error)
                self.assertIn(expected["status"], json.dumps(delivery.structured_content))

        asyncio.run(check())
        self.assertEqual(before, _hash_whitelist())

    def test_fetch_rejects_unknown_ids_and_path_traversal(self) -> None:
        async def check() -> None:
            async with Client(readonly.server) as client:
                unknown = await client.call_tool("fetch", {"id": "not-in-catalog"})
                traversal = await client.call_tool("fetch", {"id": "../../data/train.parquet"})
                self.assertTrue(unknown.is_error)
                self.assertTrue(traversal.is_error)

        asyncio.run(check())

    def test_integrity_mismatch_is_rejected(self) -> None:
        catalog = readonly._load_catalog()
        entry = dict(readonly._documents_by_id(catalog)["v62-capacity-review"])
        entry["sha256"] = "0" * 64
        with self.assertRaises(readonly.CatalogError):
            readonly._safe_document(entry)

    def test_execution_is_distinct_from_quality_acceptance(self) -> None:
        status = readonly.get_project_status()
        self.assertIn('v159-delivery', status.source_ids)
        self.assertIn('v169-platform-package-ready', status.source_ids)
        self.assertFalse(status.quality_acceptance)
        self.assertIn('V164', status.current_summary)
        self.assertIn('三目标未完成', status.current_summary)
        catalog = readonly._load_catalog()
        receipt = json.loads(readonly._document_text(readonly._documents_by_id(catalog)['v159-delivery']))
        self.assertFalse(receipt['model_promoted'])
        self.assertEqual(receipt['classifier_fits'], 6)
        self.assertEqual(receipt['full_class_gradients'], 342)
        self.assertEqual(receipt['proposals'], 574)
        self.assertEqual(receipt['updates'], 165)
        catalog = readonly._load_catalog()
        direction_id = catalog['project']['authoritative_direction_id']
        for entry in catalog['documents']:
            if entry['id'] == direction_id:
                entry['sha256'] = '0' * 64
        with patch.object(readonly, '_load_catalog', return_value=catalog):
            with self.assertRaises(readonly.CatalogError):
                readonly.get_project_status()

    def test_zero_fit_design_does_not_replace_actual_training(self) -> None:
        catalog = readonly._load_catalog()
        self.assertEqual(catalog['project']['authoritative_delivery_id'], 'v159-delivery')
        self.assertEqual(catalog['project']['authoritative_direction_id'], 'v169-platform-package-ready')
        design = json.loads(readonly._document_text(readonly._documents_by_id(catalog)['v139-design-evidence']))
        self.assertEqual((design['new_fits'], design['new_updates']), (0, 0))
        self.assertFalse(design['runtime_ready'])
        self.assertEqual(design['latest_actual_training'], 'V138')
        actual = json.loads(readonly._document_text(readonly._documents_by_id(catalog)['v155-training-receipt']))
        self.assertEqual(actual['classifier_fits'], 6)
        self.assertEqual(actual['latest_actual'], 'V155')
        self.assertFalse(actual['quality_acceptance'])
        newest = json.loads(readonly._document_text(readonly._documents_by_id(catalog)['v158-delivery']))
        self.assertEqual(newest['classifier_fits'], 60)
        self.assertEqual(newest['latest_actual'], 'V158')
        self.assertFalse(newest['quality_acceptance'])
        self.assertTrue(newest['TRAIN']['B']['all_roles_mastered'])
        self.assertEqual(newest['original_pairs']['B_vs_A']['2']['new_errors'], 10)

    def test_oof_supervision_is_not_overridden_by_fit_retention(self) -> None:
        catalog = readonly._load_catalog()
        entries = readonly._documents_by_id(catalog)
        evidence = json.loads(readonly._document_text(entries['v158-oof-class-support-replay']))
        self.assertEqual(evidence['original_OOF_role_rows'], 225614)
        self.assertTrue(evidence['deployment_FIT_retained'])
        self.assertFalse(evidence['OOF_all_classes_mastered'])
        self.assertFalse(evidence['new_formal_fit_qualification'])
        arms = [z for z in evidence['class_tradeoff_cases'] if z['arm'] == 'B']
        self.assertEqual([z['S_error_delta'] for z in arms], [50, 30, -38])
        self.assertTrue(all(z['total_CE_decreased'] for z in arms))
        self.assertLess(arms[2]['M_error_delta'], 0)
        self.assertEqual(evidence['own_fits'], 0)

    def test_v159_qualified_toy_head_does_not_activate_official_training(self) -> None:
        catalog = readonly._load_catalog()
        entries = readonly._documents_by_id(catalog)
        qualification = json.loads(readonly._document_text(entries['v159-qualification']))
        contract = json.loads(readonly._document_text(entries['v159-candidate-contract']))
        head = json.loads(readonly._document_text(entries['v159-actual-head-gradient']))
        self.assertEqual(contract['total_future_caps']['classifier_forward_chunks'], 77832)
        self.assertEqual(contract['total_future_caps']['full_class_gradients'], 2412)
        self.assertEqual(contract['allowed_activation_entries'], [])
        self.assertFalse(qualification['formal_runtime_ready'])
        self.assertFalse(qualification['official_zero_step_replay'])
        self.assertEqual(qualification['actual_official_fits'], 0)
        self.assertEqual(head['official_gradient_calls'], 0)
        self.assertTrue(head['CUDA_toy']['toy_sparse_CUDA_origin_probability_tolerance_pass'])
        self.assertLess(head['full_parameter_class_gradient_finite_difference_max_error'], 1e-9)
        self.assertEqual(contract['probability_base']['floor'], 1e-12)
        self.assertEqual(contract['probability_base']['origin_probability_tolerance'], 3e-12)

    def test_v159_execution_retention_resolves_pre_fit_conflict(self) -> None:
        entries=readonly._documents_by_id(readonly._load_catalog())
        contract=json.loads(readonly._document_text(entries['v159-execution-contract']))
        prep=json.loads(readonly._document_text(entries['v159-input-preparation']))
        self.assertEqual(contract['candidate_module'],'training/v159_current_input_boundary_v3.py')
        self.assertEqual(contract['total_future_caps']['opinion_feature_blocks'],77832)
        self.assertTrue(contract['OOF_retention_refinement']['all_accepted_deployment_correct_and_joint_scopes_unchanged'])
        self.assertTrue(prep['input_qualification_passed'])
        self.assertEqual(prep['official_classifier_calls'],0)
        self.assertEqual([r['scopes']['OOF']['unconstrained_minimum_original_errors'] for r in prep['roles']],[22,4,26])
        self.assertEqual([r['scopes']['OOF']['all_initial_correct_guard_constrained_minimum_errors'] for r in prep['roles']],[22,72,110])
        self.assertEqual(sum(r['scopes']['deployment']['protected_initial_correct_original_rows'] for r in prep['roles']),225558)

    def test_v159_current_plan_replays_actual_OOF_windows_before_calls(self) -> None:
        entries=readonly._documents_by_id(readonly._load_catalog())
        contract=json.loads(readonly._document_text(entries['v159-current-execution-contract']))
        proof=json.loads(readonly._document_text(entries['v159-protocol-rejection-cases']))
        caps=contract['total_future_caps']
        self.assertEqual((caps['classifier_forward_chunks'],caps['opinion_feature_blocks']),(78072,78072))
        self.assertEqual(caps['evaluation_classifier_forward_chunks'],720)
        self.assertEqual(caps['full_class_gradients'],2412)
        self.assertEqual(contract['activation_entries'],['training/v159_boundary_train_v3.py','training/v159_boundary_evaluate_v3.py'])
        self.assertTrue(contract['pre_registration_budget_refinement']['all_last5_actual_OOF_replay'])
        self.assertTrue(all(proof['cases'].values()))
        self.assertEqual(proof['official_classifier_calls'],0)

    def test_v159_actual_failed_preflight_costs_are_not_reset(self) -> None:
        catalog=readonly._load_catalog();entries=readonly._documents_by_id(catalog)
        self.assertEqual(catalog['project']['authoritative_direction_id'],'v169-platform-package-ready')
        proof=json.loads(readonly._document_text(entries['v159-real-repeat-vector-review']))
        self.assertEqual(proof['cumulative_actual'],dict(head_calls=124,opinion_feature_calls=124,full_class_gradients=8,fits=0,updates=0))
        self.assertFalse(proof['new_repeat_policy_active'])
        self.assertFalse(proof['first_training_issue_passed'])
        self.assertFalse(proof['quality_acceptance'])
        self.assertEqual(proof['latest_actual_training'],'V158')

    def test_v159_numeric_v4_conserves_history_and_training_caps(self) -> None:
        entries=readonly._documents_by_id(readonly._load_catalog())
        plan=json.loads(readonly._document_text(entries['v159-numeric-v4-contract']))
        cases=json.loads(readonly._document_text(entries['v159-numeric-v4-protocol-cases']))
        self.assertEqual(plan['historical_technical_cost']['classifier_forward_chunks'],124)
        self.assertEqual(plan['historical_technical_cost']['full_class_gradients'],8)
        self.assertEqual(plan['total_cumulative_caps']['classifier_forward_chunks'],78196)
        self.assertEqual(plan['total_cumulative_caps']['full_class_gradients'],2420)
        self.assertEqual(plan['total_future_caps']['fit_full_class_gradients'],2400)
        self.assertEqual(plan['total_future_caps']['fits'],6)
        self.assertEqual(plan['numeric_repeat_policy']['repeat_eps'],8)
        self.assertFalse(plan['numeric_repeat_policy']['Armijo_relaxation'])
        self.assertEqual(len(cases['cases']),20)
        self.assertTrue(all(cases['cases'].values()))

    def test_v159_real_three_role_preflight_does_not_close_training(self) -> None:
        entries=readonly._documents_by_id(readonly._load_catalog())
        proof=json.loads(readonly._document_text(entries['v159-numeric-preflight-replay-review']))
        self.assertEqual(proof['cumulative_actual'],dict(head_calls=460,opinion_feature_calls=460,full_class_gradients=20,fits=0,updates=0))
        self.assertTrue(proof['joint_TRAIN_retention']['passed'])
        self.assertFalse(proof['first_training_issue_passed'])
        self.assertFalse(proof['quality_acceptance'])
        self.assertTrue(proof['side_binding_failure']['not_relabelled_as_prospective_binding_success'])

    def test_v159_complete_delivery_failed_training_and_full_quality(self) -> None:
        entries=readonly._documents_by_id(readonly._load_catalog())
        delivery=json.loads(readonly._document_text(entries['v159-delivery']))
        data=json.loads(readonly._document_text(entries['v159-results-data']))
        self.assertEqual(delivery['classifier_fits'],6)
        self.assertEqual(delivery['ASA_errors']['B'],dict(M=1902,S=1629))
        self.assertEqual(delivery['cumulative_actual_head_calls'],14586)
        self.assertEqual(delivery['cumulative_actual_full_class_gradients'],362)
        self.assertEqual(data['arms']['B']['OOF_errors'],7030)
        self.assertEqual(data['arms']['B']['OOF_pure_current_input_errors'],6824)
        self.assertFalse(delivery['first_issue_training_qualification'])
        self.assertFalse(delivery['quality_acceptance'])
        self.assertFalse(delivery['model_promoted'])
        self.assertEqual(delivery['cached_v5_actual_evaluation_heads'],360)
        self.assertEqual(delivery['new_v7_actual_evaluation_heads'],280)

    def test_v160_completed_diagnostic_does_not_promote_failed_mechanism(self) -> None:
        entries=readonly._documents_by_id(readonly._load_catalog())
        data=json.loads(readonly._document_text(entries['v160-results-data']))
        self.assertEqual(data['actual_new_head_feature_calls'],732)
        self.assertEqual(data['actual_new_class_gradients'],6)
        self.assertEqual(data['actual_new_margin_gradients'],10)
        self.assertEqual(data['new_fits'],0)
        self.assertEqual(data['permanent_updates'],0)
        self.assertTrue(data['all_parameters_restored'])
        self.assertEqual(data['latest_completed_trained_evaluated_model'],'V159')
        self.assertFalse(data['quality_acceptance'])
        self.assertEqual(data['root_goal_status'],'active')

    def test_non_loopback_binding_requires_token(self) -> None:
        with patch.dict(os.environ, {"MCP_READONLY_TOKEN": ""}, clear=False):
            with self.assertRaises(RuntimeError):
                readonly.create_app("0.0.0.0")

    def test_http_sources_are_whitelisted_and_token_can_protect_all_routes(self) -> None:
        with patch.dict(os.environ, {"MCP_READONLY_TOKEN": ""}, clear=False):
            with TestClient(readonly.create_app("127.0.0.1")) as client:
                homepage = client.get("/")
                self.assertEqual(homepage.status_code, 200)
                self.assertEqual(homepage.json()["mode"], "read-only")
                self.assertEqual(homepage.json()["mcp_endpoint"], "/mcp")
                self.assertEqual(client.get("/health").status_code, 200)
                source = client.get("/source/v62-capacity-review")
                self.assertEqual(source.status_code, 200)
                self.assertIn("v6.2", source.text)
                self.assertEqual(client.get("/source/not-in-catalog").status_code, 404)

        with patch.dict(os.environ, {"MCP_READONLY_TOKEN": "test-secret"}, clear=False):
            with TestClient(readonly.create_app("127.0.0.1")) as client:
                self.assertEqual(client.get("/health").status_code, 401)
                authorized = client.get("/health", headers={"Authorization": "Bearer test-secret"})
                self.assertEqual(authorized.status_code, 200)


if __name__ == "__main__":
    unittest.main()
