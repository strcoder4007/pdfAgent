from dotenv import load_dotenv
import os  # Import the os module to use os.getenv


from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec


# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
api_key = os.getenv("PINECONE_API_KEY")

if not api_key:
    raise ValueError("PINECONE_API_KEY is not set in your environment variables.")

# Initialize the Pinecone client with the API key
pc = Pinecone(api_key=api_key)


pc.create_index(
  name="first",
  dimension=1024,
  metric="cosine",
  spec=ServerlessSpec(
    cloud="aws",
    region="us-east-1"
  ),
  deletion_protection="disabled"
)