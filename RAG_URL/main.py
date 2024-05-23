from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.chat_message_histories import ChatMessageHistory

from helper import transform_pretty_format
from qa_chain import get_qa_chain


if __name__ == '__main__':
	urls = [
		"",
	]
	loader = AsyncHtmlLoader(urls)
	docs = loader.load()
	html2text = Html2TextTransformer()
	docs_transformed = html2text.transform_documents(docs)
	# print(transform_pretty_format(docs_transformed))
	pretty_docs = transform_pretty_format(docs_transformed)
	print(pretty_docs)
	chat_history = ChatMessageHistory()
	qa_chain = get_qa_chain(pretty_notebook=pretty_docs)
	while True:
		user_input = input("Question: ")
		if user_input == "q":
			print("Thank you!")
			break

		ai_message = ""
		for chunk in qa_chain.stream({
			"input": user_input,
			"chat_history": chat_history.messages,
		}):
			ai_message += chunk
			print(chunk, end="", flush=True)

		print("\n")
		chat_history.add_user_message(user_input)
		chat_history.add_ai_message(ai_message)