import json, sys

f1 = json.load(open(sys.args[1]))
f2 = json.load(open(sys.args[2]))

print("\nCOMPARISON")
print(f"File 1: {sys.argv[1]}")
print(f"File 2: {sys.argv[2]}")

# Count improvements
improved1 = sum(1 for m in f1['morals'] if m.get('rag_improved'))
improved2 = sum(1 for m in f2['morals'] if m.get('rag_improved'))

print(f"\nRAG Improvements:")
print(f" File 1: {improved1}/{len(f1['morals'])} morals")
print(f" File 2: {improved2}/{len(f2['morals'])} morals")

# Show changed categories in file2
if improved2 > 0:
    print(f"\nCategories changed in File 2:")
    for m in f2['morals']:
        if m.get('rag_improved'):
            print(f"   . '{m['moral'][:40]}...")
            print(f"    {m['original_category']} -> {m['category']} (confidence: {m['rag_confidence']}%)")