import json
import asyncio
from app.api.agent import agent_chat, AgentChatRequest, ChatMessage

class MockSupabase:
    def __init__(self, model_data):
        self.model_data = model_data
        self.db = {}

    def table(self, name):
        class TableQuery:
            def __init__(self, name, parent):
                self.name = name
                self.parent = parent
            def select(self, *args, **kwargs):
                return self
            def eq(self, field, val):
                return self
            def single(self):
                class SingleResult:
                    def __init__(self, name, parent):
                        if name == "models":
                            self.data = {"weights_path": "mock/weights.json"}
                        else:
                            self.data = None
                    def execute(self):
                        return self
                return SingleResult(self.name, self.parent)
            def order(self, *args, **kwargs):
                return self
            def insert(self, data):
                self.parent.db[self.name] = data
                class Exec:
                    def execute(self):
                        return self
                return Exec()
            def execute(self):
                class ExecResult:
                    def __init__(self, data):
                        self.data = data
                if self.name == "experiments":
                    return ExecResult([{"id": "exp-1", "algorithm": "central", "status": "completed"}])
                elif self.name == "models":
                    return ExecResult([{"id": "model-1", "experiment_id": "exp-1", "algorithm": "central", "version": 1}])
                return ExecResult([])
        return TableQuery(name, self)

    class Storage:
        def __init__(self, parent):
            self.parent = parent
        def from_(self, bucket):
            class Bucket:
                def __init__(self, parent):
                    self.parent = parent
                def download(self, path):
                    return json.dumps(self.parent.model_data).encode()
            return Bucket(self.parent)

    @property
    def storage(self):
        return self.Storage(self)

def test_agent_chat_flow():
    model_data = {
        "weights": [0.5, -0.2],
        "bias": 0.1,
        "scalers": [{"min": 0.0, "range": 10.0}, {"min": 0.0, "range": 1.0}],
        "feature_names": ["sl.no", "Age"],
        "encoders": {}
    }

    mock_sb = MockSupabase(model_data)

    import app.api.agent as agent_module
    import app.api.predict as predict_module

    old_agent_sb = agent_module.get_supabase
    old_predict_sb = predict_module.get_supabase
    agent_module.get_supabase = lambda: mock_sb
    predict_module.get_supabase = lambda: mock_sb

    try:
        user = {"id": "user-123"}
        loop = asyncio.get_event_loop()

        # 1. Welcome and choose model
        req = AgentChatRequest(
            model_id=None,
            messages=[ChatMessage(role="user", content="hello")]
        )
        res = loop.run_until_complete(agent_chat(req, user=user))
        assert "select one of the models" in res["message"]
        assert len(res["options"]) == 1
        assert res["options"][0]["value"] == "model-1"

        # 2. Start collecting features
        req = AgentChatRequest(
            model_id="model-1",
            messages=[ChatMessage(role="user", content="hello")],
            current_feature=None,
            collected_features={}
        )
        res = loop.run_until_complete(agent_chat(req, user=user))
        assert "sl.no" in res["message"]
        assert res["current_feature"] == "sl.no"

        # 3. Enter sl.no (numerical value) -> Ask for Age
        req = AgentChatRequest(
            model_id="model-1",
            messages=[
                ChatMessage(role="assistant", content="What is the value for sl.no?"),
                ChatMessage(role="user", content="1")
            ],
            current_feature="sl.no",
            collected_features={}
        )
        res = loop.run_until_complete(agent_chat(req, user=user))
        assert "Age" in res["message"]
        assert res["current_feature"] == "Age"
        assert res["collected_features"]["sl.no"] == 1.0

        # 4. Enter Age (numerical value) -> Prediction complete
        req = AgentChatRequest(
            model_id="model-1",
            messages=[
                ChatMessage(role="assistant", content="What is the value for Age?"),
                ChatMessage(role="user", content="45")
            ],
            current_feature="Age",
            collected_features={"sl.no": 1.0}
        )
        res = loop.run_until_complete(agent_chat(req, user=user))
        assert "Prediction Result" in res["message"]
        assert res["prediction"] is not None
        assert res["prediction"]["class_label"] in ["Positive", "Negative"]

    finally:
        agent_module.get_supabase = old_agent_sb
        predict_module.get_supabase = old_predict_sb
