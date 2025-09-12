from utils import vs_utils

def main():
    """Main indexing function."""
    print("Simple Indexer")
    print("=" * 40)
    
    shoe_image_links = vs_utils.get_shoe_image_links()
    print(f"Processing {len(shoe_image_links)} shoe images...")
    
    vs_utils.process_images_to_json_and_insert(shoe_image_links)
    print("\n Testing search...")
    try:
        results = vs_utils.vector_search_shoes("spor ayakkabı", 2)
        if results:
            print("Search works!")
        else:
            print("Search test failed")
    except Exception as e:
        print(f"Test error: {e}")
    
    print("\n Ready!")

if __name__ == "__main__":
    main()


