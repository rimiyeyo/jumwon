import json
from typing import List

from dotenv import load_dotenv
from langchain.embeddings import CacheBackedEmbeddings
from langchain.globals import set_debug
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_upstage import UpstageEmbeddings
from pydantic import BaseModel, Field

set_debug(True)
load_dotenv()
# logging.langsmith("Module")

# 모델 정의
gpt4o_mini = ChatOpenAI(model_name="gpt-4o-mini-2024-07-18", temperature=0.3)
gpt4o = ChatOpenAI(model_name="gpt-4o-2024-08-06") 
claude = ChatAnthropic(model_name="claude-3-5-sonnet-20240620")

# 데이터 로드
file_dir = "./menu_1017.json"
data = json.load(open(file_dir, "r", encoding="utf-8"))


class CandidateMenus(BaseModel):
    menus: List[str] = Field(description="후보 메뉴 이름들")


class OrderSingleItem(BaseModel):
    name: str = Field(description="주문하려는 음식/메뉴 이름")
    price: int = Field(description="주문하려는 음식/메뉴의 가격")
    quantity: int = Field(description="주문하려는 음식/메뉴의 수량")


class OrderSetItem(BaseModel):
    quantity: int = Field(description="주문하려는 세트의 수량")
    price: int = Field(description="주문하려는 세트의 합계 가격 (세트 할인 적용)")
    items: List[OrderSingleItem] = Field(description="주문하려는 세트의 구성 품목")


OrderItem = OrderSingleItem | OrderSetItem


class OrderResponse(BaseModel):
    """
    에이전트가 사용자 요청에 따라 채운 주문 정보
    """

    completion: bool = Field(
        description="현재까지 사용자 요청에 있는 정보로 주문 정보(품목, 수량)가 충족되어 주문 가능한 상태면 true, 아니면 false")
    message: str = Field(description="에이전트가 사용자에게 주문 요청에 대한 답변 메시지")
    order: List[OrderItem] = Field(description="사용자의 요청에 따라 만든 주문 목록, 품목은 세트 혹은 단품 상품이 복합적으로 여러개 존재할 수 있음")


class ShoppingCart:
    def __init__(self):
        self.cart: List[OrderItem] = []
    
    def add_to_cart(self, new_items: List[OrderItem]) -> List[OrderItem]:
        self.cart += new_items
        return self.cart
    
    def get_total_price(self) -> int:
        return sum([order.price * order.quantity for order in self.cart])

    def get_order_message(self) -> str:
        total_price = self.get_total_price()
        order_details: List[str] = []
        
        for order in self.cart:
            names = [item.name for item in order.items] if order is OrderSetItem else [order.name]
            item_type = "세트" if order is OrderSetItem else "단품"
            item_name = ", ".join(names)
            item_quantity = order.quantity
            item_price = order.price

            template = """
            {type} 유형으로
            {names}를
            {quantity}개를
            {price}원""".strip()
            
            order_details.append(
                template.format(
                    type=item_type,
                    names=item_name,
                    quantity=item_quantity,
                    price=item_price,
                )
            )
        order_string = "\n=====\n".join(order_details)
        return f"[주문항목]\n=====\n{order_string}\n\n[총 금액]\n\n{total_price}원"


class IntentChain:
    def __init__(self):
        self.model = gpt4o_mini

    def additional_invoke(self, question):
        intent_chain = ChatPromptTemplate([
            ("system", """
            사용자의 질문을 '종료', '결제', '추천' 중 하나로 분류하세요.

            분류 기준:
            - 종료: 주문을 종료하려는 경우
            - 결제: 주문 완료 후 결제를 요청하는 경우, 요청사항이 없다고 하는 경우 (예: '결제할거야', '주문 완료', '없어')
            - 추천: 취소나 결제가 아닌 기타 문의 (예: '추천 메뉴 있어요?', '가장 인기 있는 메뉴는?','아까 주문한거 취소할래')

            출력 형식:
            - 예시: 결제
            """),
            ("ai","추가주문이나 다른 요청이 있으신가요?"),
            ("human", "{input}")
        ]) | {"input": RunnablePassthrough()} | self.model | StrOutputParser()
        
        return intent_chain.invoke(question)


class RecommendModule:
    def __init__(self, model, chat_history):
        self.model = model
        self.chat_history = chat_history
    
    @property
    def menu_chain(self):
        parser = PydanticOutputParser(pydantic_object=CandidateMenus)
        guess_template = ChatPromptTemplate(
            [
                (
                    "system", 
                    """
                    이전 대화 내역을 참고하여 사용자가 언급한 메뉴를 검색할 수 있도록 필요한 정보를 추출하세요. 
                    사용자가 비교 요청 또는 추가 정보를 요구한 메뉴 이름을 "검색_내용" 항목에 포함시키세요.
                    사용자가 주문한다고 언급한 모든 메뉴 항목을 반드시 "검색_내용"에 포함시키세요.
                    검색이 필요한 항목과 주문 내역 외에는 "검색_내용"에 포함시키지 마세요.

                    [OUTPUT_FORMAT]
                    {instruction}

                    반드시 위의 출력 형식에 맞춰 JSON 형태로 응답해주세요. 마크다운 표시는 하지 마세요.
                    """
                ),
                *self.chat_history.messages,
                 ("human", "{input}"),
            ],
            partial_variables={"instruction": parser.get_format_instructions()},
        )

        guess_chain = {"input": RunnablePassthrough()} | guess_template | gpt4o_mini | parser
        return guess_chain
    
    @property
    def recommend_chain(self):
        docs = [
            Document(
                page_content=json.dumps(obj["page_content"], ensure_ascii=False),
            )
            for obj in json.load(open(file_dir, "r", encoding="utf-8"))
        ]
        text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=100, chunk_overlap=0)
        split_docs = text_splitter.split_documents(docs)

        embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
        cache_dir = LocalFileStore(f"./.cache/embeddings/{file_dir.split('/')[-1]}")
        cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings=embeddings,
            document_embedding_cache=cache_dir,
            namespace="solar-embedding-1-large",
        )
        vectorstore = FAISS.from_documents(split_docs, cached_embedder)
        faiss = vectorstore.as_retriever(search_kwargs={"k": 4})

        bm25 = BM25Retriever.from_documents(split_docs)
        bm25.k = 2

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25, faiss],
            weights=[0.3, 0.7],
            search_type="mmr",
        )
        
        parser = PydanticOutputParser(pydantic_object=OrderResponse)
        recommend_template = ChatPromptTemplate(
            [
                (
                    "system",
                    """
                    당신은 맥도날드의 친절한 점원입니다.
                    고객님의 주문을 도와드리세요. 나이와 상관없이 '고객님'이라고만 부르고, 모든 설명은 어린아이도 이해할 수 있게 해주세요. 
                    대답은 세 문장 이내로, 간결하고 친절하게 응대합니다.
                    
                    **응대 지침:**
                    - 질문과 가장 관련성이 높은 정보를 선택해 대답하세요.
                    - 메뉴 추천은 2개 이하로 제한하며, 신메뉴를 우선 추천하세요.
                    - 확실하지 않으면 "정확한 답변을 드리기 어렵습니다만, 추가로 확인 후 도와드리겠습니다."라고 답하세요.
                    - 이전 대화내역을 최우선으로 고려하고, 그 외 현재 대화의 [CONTEXT]를 참고합니다.

                    **주문 조건:**
                    - 메뉴는 세트와 단품으로 구분됩니다.
                    - 세트 메뉴는 기본으로 버거, 사이드, 음료가 포함되며, 미디엄 사이즈가 기본입니다.
                    - 미디엄 사이즈 세트 메뉴 가격은 [CONTEXT]의 'price' 가격입니다. 해당 정보가 없으면 "죄송합니다, 세트 구성이 불가능한 항목입니다. 대신 단품으로 주문하시거나 다른 메뉴를 선택해 주세요."라고 안내하세요.
                    - 라지 사이즈 세트 메뉴로의 업그레이드는 800원 추가 요금이 부과됩니다.
                    - 사이드는 기본으로 후렌치 후라이 미디엄입니다.
                    - 음료는 기본으로 코카콜라 미디엄이며, 음료 변경은 [코카콜라, 코카콜라 제로, 스프라이트, 환타]중에 가능합니다.
                    
                    [CONTEXT]
                    {context}
                    """
                ),
                *self.chat_history.messages,
                (
                    "human",
                    """
                    [QUERY]
                    {input}

                    [OUTPUT_FORMAT]
                    {instruction}
                    """
                ),
            ],
            partial_variables={"instruction": parser.get_format_instructions()},
        )

        recommend_chain = {
            "context": ensemble_retriever,
            "input": RunnablePassthrough(),
        } | recommend_template | self.model | parser
        return recommend_chain

    def invoke(self, user_query: str) -> OrderResponse:
        candidate_menus: CandidateMenus = self.menu_chain.invoke(user_query)
        order_request = f"[사용자 요청]\n{user_query}\n\n[후보 메뉴 이름들]\n{', '.join(candidate_menus.menus)}"
        return self.recommend_chain.invoke(order_request)

# 주문 모듈 클래스
class OrderModule: 
    def __init__(self, model):
        self.cart = ShoppingCart()
        self.intent_chain = IntentChain()
        self.chat_history = ChatMessageHistory()
        self.recommend_module = RecommendModule(model, self.chat_history)
        
    def handle_additional_requests(self):
        while True:
            print("add_req")
            user_message = input("추가 주문이나 다른 요청이 있으신가요?").strip()
            
            intent = self.intent_chain.additional_invoke(user_message)
            print(f"intent:{intent}")
            print(f"type:{type(intent)}")
            
            if intent == "추천":
                self.execute_additional_order(user_message)  # 추가 주문 처리
            elif intent == "결제":
                cart_menu = ShoppingCart.print_order()
                print(cart_menu['message']) #이 값을 서버로 가져가시면 됩니다.
                print("결제를 도와드리겠습니다")
                # 결제 로직 추가 필요
                break
            elif intent == "종료":
                print("주문을 종료합니다. 다음에 또 뵙겠습니다 고객님.")
                break
            else:
                print("죄송합니다. 요청을 이해하지 못했습니다.")

    def handle_prompt(self, user_query: str) -> str:
        order_response: OrderResponse = self.recommend_module.invoke(user_query)
        self.cart.add_to_cart(order_response.order)
        self.chat_history.add_user_message(user_query)
        self.chat_history.add_ai_message(order_response.message)
        return order_response.message


# 메인 실행 부분
if __name__ == "__main__":
    order_module = OrderModule(gpt4o)
    try:
        user_message = input("입력 :")
        order_module.handle_prompt(user_message)
    except Exception as e:
        print(f"예상치 못한 오류가 발생했습니다: {e}")
