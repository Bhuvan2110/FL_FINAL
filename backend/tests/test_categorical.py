import json
from app.tasks.celery_app import run_training_task

class MockSupabase:
    def __init__(self, dataset_bytes):
        self.dataset_bytes = dataset_bytes
        self.db = {}
        self.storage_data = {}

    def table(self, name):
        class TableQuery:
            def __init__(self, name, parent):
                self.name = name
                self.parent = parent
            def select(self, *args, **kwargs):
                return self
            def eq(self, field, val):
                self.field = field
                self.val = val
                return self
            def single(self):
                class SingleResult:
                    def __init__(self, name, parent, val):
                        if name == "datasets":
                            self.data = {"storage_path": "datasets/test.csv"}
                        elif name == "models":
                            self.data = {"weights_path": "models/test-experiment/model_v1.json"}
                        else:
                            self.data = None
                    def execute(self):
                        return self
                return SingleResult(self.name, self.parent, self.val)
            def update(self, data):
                class UpdateQuery:
                    def __init__(self, name, parent, data):
                        self.name = name
                        self.parent = parent
                        self.data = data
                    def eq(self, field, val):
                        self.parent.db[self.name] = (field, val, self.data)
                        class Exec:
                            def execute(self):
                                return self
                        return Exec()
                return UpdateQuery(self.name, self.parent, data)
            def insert(self, data):
                self.parent.db[self.name] = data
                class Exec:
                    def execute(self):
                        class Result:
                            def __init__(self, data):
                                self.data = [data]
                        return Result(data)
                return Exec()
        return TableQuery(name, self)

    class Storage:
        def __init__(self, parent):
            self.parent = parent
        def from_(self, bucket):
            class Bucket:
                def __init__(self, bucket, parent):
                    self.bucket = bucket
                    self.parent = parent
                def download(self, path):
                    if self.bucket == "datasets":
                        return self.parent.dataset_bytes
                    elif self.bucket == "models":
                        return self.parent.storage_data[path]
                def upload(self, path, data, mime):
                    self.parent.storage_data[path] = data
                    return True
            return Bucket(bucket, self.parent)

    @property
    def storage(self):
        return self.Storage(self)

def test_categorical_training_and_prediction():
    # CSV with categorical columns Sex, ChestPainType and target HeartDisease
    csv_content = (
        "Age,Sex,ChestPainType,HeartDisease\n"
        "40,M,ATA,0\n"
        "49,F,NAP,1\n"
        "37,M,NAP,0\n"
        "48,F,ASY,1\n"
        "54,M,TA,0\n"
    ).encode()

    mock_sb = MockSupabase(csv_content)
    
    # Mock supabase client get_supabase in both sc and ap
    import app.db.supabase_client as sc
    import app.api.predict as ap
    
    old_sc_sb = sc.get_supabase
    old_ap_sb = ap.get_supabase
    sc.get_supabase = lambda: mock_sb
    ap.get_supabase = lambda: mock_sb
    
    try:
        class DummyCeleryTask:
            class DummyRequest:
                def __init__(self):
                    self.id = "test-task-123"
            def __init__(self):
                self.request = self.DummyRequest()

        config = {
            "dataset_id": "test-dataset",
            "algorithm": "central",
            "n_rounds": 1,
            "lr": 0.1,
            "n_clients": 2,
            "local_epochs": 1,
            "mu": 0.1,
            "clip_norm": 1.0,
            "noise_multiplier": 1.0,
            "delta": 1e-5,
            "non_iid": False,
            "user_id": "mock-user"
        }

        # Train model
        run_training_task.run.__func__(DummyCeleryTask(), "test-experiment", config)
        
        # Verify model data was saved with encoders
        model_path = "models/test-experiment/model_v1.json"
        assert model_path in mock_sb.storage_data
        model_data = json.loads(mock_sb.storage_data[model_path].decode())
        
        assert "encoders" in model_data
        assert model_data["encoders"]["Sex"] == ["F", "M"]
        assert model_data["encoders"]["ChestPainType"] == ["ASY", "ATA", "NAP", "TA"]
        
        # Test predictions using predict endpoints
        from app.api.predict import predict_single, SinglePredictRequest
        
        # M should map to 1.0, ATA to 1.0
        req = SinglePredictRequest(
            model_id="test-model",
            features={"Age": 40, "Sex": "M", "ChestPainType": "ATA"}
        )
        
        user = {"id": "user-123"}
        
        # single predict
        import asyncio
        res = asyncio.run(predict_single(req, user=user))
        assert "prediction" in res
        
    finally:
        sc.get_supabase = old_sc_sb
        ap.get_supabase = old_ap_sb
