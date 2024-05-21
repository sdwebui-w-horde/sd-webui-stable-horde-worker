from typing import List, Optional
import requests


class HordeWorker:
    id: str
    name: str
    maintenance_mode: bool

    def __init__(self, session: requests.Session, worker_id: str):
        # https://stablehorde.net/api/#operations-v2-get_worker_single
        r = session.get(f"/api/v2/workers/{worker_id}")
        json = r.json()
        self.id = worker_id
        self.name = json["name"]
        self.maintenance_mode = json["maintenance_mode"]


class HordeUser:
    id: str
    username: str
    kudos: int
    workers: List[HordeWorker]

    def __init__(self, session: requests.Session):
        # https://stablehorde.net/api/#operations-v2-get_find_user
        r = session.get("/api/v2/find_user")
        json = r.json()
        workers = []

        for worker in json["worker_ids"]:
            workers.append(HordeWorker(session, worker))

            self.id = json["id"]
            self.username = json["username"]
            self.kudos = json["kudos"]
            self.workers = workers

    def get_worker(self, worker_id: str) -> Optional[HordeWorker]:
        for worker in self.workers:
            if worker.id == worker_id:
                return worker
