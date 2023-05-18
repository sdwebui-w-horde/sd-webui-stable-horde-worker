from typing import List
import aiohttp


class HordeWorker:
    id: str
    name: str
    maintenance_mode: bool

    @classmethod
    async def get(cls, session: aiohttp.ClientSession, worker_id: str):
        # https://stablehorde.net/api/#operations-v2-get_worker_single
        r = await session.get(f"/api/v2/workers/{worker_id}")
        json = await r.json()
        return HordeWorker(worker_id, json["name"], json["maintenance_mode"])

    def __init__(self, id: str, name: str, maintenance_mode: bool):
        self.id = id
        self.name = name
        self.maintenance_mode = maintenance_mode


class HordeUser:
    id: str
    username: str
    kudos: int
    workers: List[HordeWorker]

    @classmethod
    async def get(cls, session: aiohttp.ClientSession):
        # https://stablehorde.net/api/#operations-v2-get_find_user
        r = await session.get("/api/v2/find_user")
        json = await r.json()
        workers = []

        for worker in json["worker_ids"]:
            workers.append(await HordeWorker.get(session, worker))

        return HordeUser(
            json["id"], json["username"], json["kudos"], workers
        )

    def __init__(self, id: str, username: str, kudos: int, workers: List[HordeWorker]):
        self.id = id
        self.username = username
        self.kudos = kudos
        self.workers = workers
