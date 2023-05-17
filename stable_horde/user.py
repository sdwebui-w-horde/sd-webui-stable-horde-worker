from typing import List
import aiohttp

class HordeUser:
    id: str
    username: str
    kudos: int
    workers: List[str]

    @classmethod
    async def get(session: aiohttp.ClientSession):
        # https://stablehorde.net/api/#operations-v2-get_find_user
        r = await session.get('/v2/find_user')
        json = await r.json()
        return HordeUser(json["id"], json["username"], json["kudos"], json["worker_ids"])

    def __init__(self, id: str, username: str, kudos: int, workers: List[str]):
        self.id = id
        self.username = username
        self.kudos = kudos
        self.workers = workers


class HordeWorker:
    id: str
    maintenance_mode: bool

    @classmethod
    async def get(session: aiohttp.ClientSession, worker_id: str):
        # https://stablehorde.net/api/#operations-v2-get_worker_single
        r = await session.get(f'/v2/workers/{worker_id}')
        json = await r.json()
        return HordeWorker(worker_id, json["maintenance_mode"])
    
    def __init__(self, id: str, maintenance_mode: bool):
        self.id = id
        self.maintenance_mode = maintenance_mode
