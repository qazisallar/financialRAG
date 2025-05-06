# 1. First install udocker
# !udocker --allow-root install

# # 2. Kill existing processes and clean up
# !pkill -9 -f postgres
# !rm -rf /content/pgdata
# !udocker --allow-root rm pgvector
# !rm -f postgres.log

# 3. Create fresh directory
!mkdir -p /content/pgdata
!chmod -R 777 /content/pgdata

# 4. Pull and create container with correct image path
!docker --allow-root pull ankane/pgvector
!docker --allow-root create --name=pgvector ankane/pgvector

# 5. Run the container
!nohup docker --allow-root run \
    --env="POSTGRES_DB=ai" \
    --env="POSTGRES_USER=ai" \
    --env="POSTGRES_PASSWORD=ai" \
    --env="PGDATA=/var/lib/postgresql/data/pgdata" \
    --volume="/content/pgdata:/var/lib/postgresql/data" \
    --publish="5532:5432" \
    pgvector > postgres.log 2>&1 &

# 6. Connection testing
import time
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

def test_db_connection(max_retries=5, wait_time=10):
    db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

    for attempt in range(max_retries):
        try:
            print(f"\nConnection attempt {attempt + 1}/{max_retries}")
            engine = create_engine(db_url)
            with engine.connect() as connection:
                result = connection.execute(text("SELECT version();"))
                version = result.fetchone()[0]
                print("✅ Successfully connected to PostgreSQL!")
                print(f"Server Version: {version}")

                # Test vector extension
                connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                print("✅ Vector extension ready!")
                return engine
        except OperationalError as e:
            print(f"Attempt {attempt + 1} failed, waiting {wait_time} seconds...")
            print("\nChecking postgres status:")
            !ps aux | grep postgres
            print("\nLatest logs:")
            !tail -n 20 postgres.log
            time.sleep(wait_time)

    return None

# 7. Apply 30 secs sleep time to wait for DB to finish set up before testing for connection
print("Waiting for database to initialize...")
time.sleep(30)

engine = test_db_connection()

if engine:
    print("\n✅ Database is ready for RAG Agent initialization!")
else:
    print("\n❌ Database connection failed. Please check the logs above.")