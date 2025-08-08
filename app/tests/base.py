
import datetime, json
from ..core.utils import ensure_dir

class TestResult:
    def __init__(self, name, params=None, metrics=None, data=None, files=None):
        self.test = name
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.params = params or {}
        self.metrics = metrics or {}
        self.data = data or {}
        self.files = files or {}

    def to_dict(self):
        return {
            "test": self.test,
            "timestamp": self.timestamp,
            "params": self.params,
            "metrics": self.metrics,
            "data": self.data,
            "files": self.files,
        }

    def export_json(self, out_dir, base_name=None):
        ensure_dir(out_dir)
        if base_name is None:
            base_name = f"{self.test}_{self.timestamp.replace(':','').replace(' ','_')}"
        path = f"{out_dir}/{base_name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return path
