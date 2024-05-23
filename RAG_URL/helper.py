from typing import List, Sequence, Union

from langchain_core.documents import Document


def transform_pretty_format(
	docs: Union[List[Document], Sequence[Document]]
) -> str:
	pretty_document = "<documents>"
	for index, doc in enumerate(docs, start=1):
		pretty_document += f"<document {index}>\n"
		pretty_document += f"<title>\n{doc.metadata['title']}\n</title>\n"
		pretty_document += f"<content>\n{doc.page_content}\n<content>\n"
		pretty_document += f"</document>\n\n"

	pretty_document += "</documents>"
	return pretty_document
