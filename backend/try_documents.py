from rag import document_store
print("location",document_store.location())
print("saved at",document_store.save("../uploads/aldermoor_procurement_policy.pdf","aldermoor_procurement_policy.pdf"))
with document_store.pdfs() as files:
    for local_path,source in files:
        print("stored:",source)