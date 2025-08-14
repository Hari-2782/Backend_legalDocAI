#!/usr/bin/env python3
"""
Script to restore connection to existing ChromaDB Cloud collection
"""

import chromadb
import os
import sys

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.config import settings

def restore_chroma_connection():
    """Restore connection to existing ChromaDB Cloud collection"""
    try:
        print("=== Restoring ChromaDB Cloud Connection ===\n")
        
        # Set environment variables for ChromaDB Cloud
        os.environ["CHROMA_SERVER_HOST"] = settings.CHROMA_CLOUD_HOST
        os.environ["CHROMA_SERVER_HTTP_PORT"] = "443"
        os.environ["CHROMA_SERVER_SSL_ENABLED"] = "true"
        os.environ["CHROMA_AUTH_CREDENTIALS"] = settings.CHROMA_CLOUD_API_KEY
        
        print(f"Connecting to: {settings.CHROMA_CLOUD_HOST}")
        print(f"API Key: {settings.CHROMA_CLOUD_API_KEY[:20]}...")
        
        # Create client
        client = chromadb.Client()
        print("✅ ChromaDB Cloud client created")
        
        # List all collections to see what's available
        print("\n1. Listing all available collections...")
        try:
            collections = client.list_collections()
            print(f"Found {len(collections)} collections:")
            for col in collections:
                print(f"  - {col.name} (ID: {col.id})")
                try:
                    count = col.count()
                    print(f"    Documents: {count}")
                except Exception as e:
                    print(f"    Could not count: {e}")
        except Exception as e:
            print(f"❌ Could not list collections: {e}")
            return False
        
        # Try to get the legal_chunks collection
        print("\n2. Connecting to legal_chunks collection...")
        try:
            collection = client.get_collection(name="legal_chunks")
            print("✅ Successfully connected to legal_chunks collection")
            
            # Count documents
            count = collection.count()
            print(f"📊 Total documents: {count}")
            
            if count > 0:
                print("🎉 Collection contains data!")
                
                # Try a simple query to verify data
                print("\n3. Testing query functionality...")
                try:
                    results = collection.query(
                        query_texts=["contract terms"],
                        n_results=3
                    )
                    
                    if results["documents"] and results["documents"][0]:
                        print(f"✅ Query successful! Found {len(results['documents'][0])} results")
                        
                        # Show first result
                        first_doc = results["documents"][0][0]
                        first_meta = results["metadatas"][0][0] if results["metadatas"] and results["metadatas"][0] else {}
                        
                        print(f"\nFirst result:")
                        print(f"Text: {first_doc[:200]}...")
                        print(f"Metadata: {first_meta}")
                        
                        return True
                    else:
                        print("⚠ Query returned no results")
                        return False
                        
                except Exception as e:
                    print(f"❌ Query failed: {e}")
                    return False
            else:
                print("⚠ Collection is empty - this suggests a connection issue")
                return False
                
        except Exception as e:
            print(f"❌ Could not connect to legal_chunks collection: {e}")
            print("This collection might not exist or there's a connection issue")
            return False
            
    except Exception as e:
        print(f"❌ Script failed: {e}")
        return False

def test_specific_file_query(file_hash: str):
    """Test querying for a specific file"""
    try:
        print(f"\n=== Testing Query for File: {file_hash} ===\n")
        
        # Set environment variables
        os.environ["CHROMA_SERVER_HOST"] = settings.CHROMA_CLOUD_HOST
        os.environ["CHROMA_SERVER_HTTP_PORT"] = "443"
        os.environ["CHROMA_SERVER_SSL_ENABLED"] = "true"
        os.environ["CHROMA_AUTH_CREDENTIALS"] = settings.CHROMA_CLOUD_API_KEY
        
        client = chromadb.Client()
        collection = client.get_collection(name="legal_chunks")
        
        # Query with file filter
        print("Querying for specific file...")
        results = collection.query(
            query_texts=["contract"],
            n_results=10,
            where={"file_hash": file_hash}
        )
        
        if results["documents"] and results["documents"][0]:
            print(f"✅ Found {len(results['documents'][0])} chunks for file {file_hash}")
            
            # Show chunk details
            for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
                print(f"\nChunk {i+1}:")
                print(f"  Page: {meta.get('page', 'N/A')}")
                print(f"  Length: {len(doc)} chars")
                print(f"  Preview: {doc[:100]}...")
            
            return True
        else:
            print("⚠ No chunks found for this file")
            return False
            
    except Exception as e:
        print(f"❌ File query failed: {e}")
        return False

if __name__ == "__main__":
    print("=== ChromaDB Cloud Connection Restore ===\n")
    
    # Test basic connection
    connection_ok = restore_chroma_connection()
    
    if connection_ok:
        print("\n✅ ChromaDB Cloud connection restored successfully!")
        
        # Test specific file query
        file_hash = "8bceeb2b84f0283f5273b0330a16b5b2725bfcb2792db6bd9fa1ccf8d4336c52"
        test_specific_file_query(file_hash)
        
        print("\n🎉 You can now restart your server and it should work!")
    else:
        print("\n❌ ChromaDB Cloud connection could not be restored")
        print("Please check your configuration and try again")
