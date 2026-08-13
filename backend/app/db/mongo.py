from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings

# This one client is created once and reused everywhere —
# same reasoning as why the LangGraph graph is compiled once at startup,
# not rebuilt on every request.
mongo_client: AsyncIOMotorClient | None = None


def connect_to_mongo() -> AsyncIOMotorClient:
    """Create the Motor client. Called once, from FastAPI's lifespan startup."""
    global mongo_client
    mongo_client = AsyncIOMotorClient(settings.mongo_uri)
    return mongo_client


def close_mongo_connection() -> None:
    """Cleanly close the connection. Called once, from lifespan shutdown."""
    if mongo_client is not None:
        mongo_client.close()


def get_database():
    """Points to the 'vaultmind' database inside the cluster."""
    return mongo_client[settings.mongo_db_name]


def get_users_collection():
    return get_database()["users"]


def get_conversations_collection():
    return get_database()["conversations"]


async def ping_mongo() -> bool:
    """Health check — same idea as your deep /health probe for the LangGraph graph."""
    try:
        await mongo_client.admin.command("ping")
        return True
    except Exception:
        return False