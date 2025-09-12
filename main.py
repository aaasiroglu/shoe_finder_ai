import json
import chainlit as cl
from typing import Optional, List
from openai import AzureOpenAI
from config import settings 
from utils import vs_utils

@cl.step(type="tool")
async def search_shoes(query: str) -> str:
    return vs_utils.vector_search_shoes(query, 5)

@cl.step(type="tool")
async def get_weather(city: str) -> str:
    return vs_utils.get_current_weather(city)

@cl.step(type="tool")
async def plan_and_search(user_request: str) -> str:
    plan = await vs_utils.extract_shoe_intent(user_request)
    intent_details = f"""**Kişisel Analiz Tamamlandı**

 **Size Özel Arama Kriterleri:**
-  **Aradığınız Tip:** {plan.get('shoe_type', 'Açık tercih belirtilmedi')}
-  **Renk Tercihi:** {plan.get('color', 'Renk tercihi esnek')}
-  **Stil Yaklaşımı:** {plan.get('style', 'Stil tercihi geniş')}
-  **Kullanım Amacı:** {plan.get('use_case', 'Genel kullanım')}"""

    if plan.get('special_features'):
        intent_details += f"\n-  **Özel İstekleriniz:** {', '.join(plan['special_features'])}"
        
    intent_details += f"\n\n **Arama Sorgusu:** `{plan.get('query', user_request)}`"
    intent_details += "\n\n **Size özel seçilmiş ayakkabılar yükleniyor...**\n\n"
    
    results = await search_shoes(plan["query"])
    if "bulunamadı" in results or "bulunmuyor" in results:
        return intent_details + f" **Üzgünüm!** Tam bu kriterlere uygun ayakkabı bulamadım.\n\n🤔 **Alternatif Öneriler:** Lütfen:\n- Renk tercihlerinizi genişletin\n- Farklı stillerle ilgili açık olun\n- Benzer özelliklerdeki ayakkabıları değerlendirin\n\n{results}"
    
    return intent_details + results + "\n\n💎 **Bu seçenekler arasından hangisi daha çok ilginizi çekiyor? Daha detaylı bilgi veya alternatifler istiyorsanız sormaktan çekinmeyin!**"


tools = [
            {
            "type": "function",
            "function": {
                "name": "plan_and_search",
                "description": "Detaylı kullanıcı taleplerini analiz eder ve ayakkabı veritabanında aramak için yapılandırılmış bir sorgu oluşturur. Stil, amaç, renk gibi birden çok kriter içeren talepler için kullanılır.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_request": {
                            "type": "string",
                            "description": "Kullanıcının aradığı ayakkabıyı tanımlayan doğal dil sorgusu. Örnek: 'düğün için şık topuklu ayakkabı'."
                        }
                    },
                    "required": ["user_request"]
                }
            }
        },
            {
            "type": "function",
            "function": {
                "name": "search_shoes",
                "description": "Veritabanında doğrudan ayakkabı araması yapar. 'siyah bot' gibi net ve basit sorgular için kullanılır.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Veritabanında aranacak net sorgu metni."
                        }
                    },
                    "required": ["query"]
                }
            }
        },
            {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Belirtilen bir şehir için güncel hava durumu bilgisini alır. Mevsimlik ayakkabı önerileri için kullanılabilir.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "Hava durumu bilgisi alınacak şehrin adı."
                        }
                    },
                    "required": ["city"]
                }
            }
        }
    ]

client = AzureOpenAI(
    api_key=settings.openai_key,
    api_version="2024-02-15-preview",
    azure_endpoint=settings.openai_endpoint,
    default_headers={"Ocp-Apim-Subscription-Key": settings.openai_key}
)

async def call_azure_openai(messages):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            stream=False,
            temperature=0.0,           
            top_p=1,
            max_tokens=1024,
            presence_penalty=0,
            frequency_penalty=0,
            seed=42,                 
            tools=tools,
        )
        message = response.choices[0].message

        if message.tool_calls:
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                if function_name == "get_weather":
                    tool_result = await get_weather(function_args["city"])
                elif function_name == "search_shoes":
                    tool_result = await search_shoes(function_args["query"])
                elif function_name == "plan_and_search":
                    tool_result = await plan_and_search(function_args["user_request"])
                else:
                    tool_result = f"Tool {function_name} not found"
                
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call]
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(tool_result)
                })
            
            final_response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                stream=False,
                temperature=0,
                top_p=1,
                max_tokens=800,
                presence_penalty=0,
                frequency_penalty=0,
                seed=42,            
                tools=tools,
            )
            return final_response.choices[0].message.content
        
        return message.content

    except Exception as e:
        print(f"Error calling Azure OpenAI: {str(e)}")
        return "I apologize, but I encountered an error. Please try again."

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("messages", [
        {"role": "system", "content": settings.saler_system_prompt},
    ])
    
    
@cl.on_message
async def on_message(message: cl.Message):
    thinking = cl.Message(content="Sizin için en uygun ayakkabıları arıyorum...", author="Ayakkabı Asistanı 👟")
    await thinking.send()
    messages = cl.user_session.get("messages")
    messages.append({"role": "user", "content": message.content})
    
    try:
        content = await call_azure_openai(messages)
        
        if "bulundu" in content:
            content += "\n\n**Kişisel Önerim:** Bu seçenekler arasından hangisi size daha yakın geliyor? Daha fazla detay veya alternatif istiyorsanız sormaktan çekinmeyin!"
        elif "bulunamadı" in content or "bulunmuyor" in content:
            content += "\n\n**Alternatif Çözüm:** İstediklerinize benzer farklı seçenekleri araştırabilirim. Tercihlerinizi biraz daha detaylandırabilir misiniz?"
            
    except Exception as e:
        content = "Üzgünüm, şu anda teknik bir sorun yaşıyoruz. Lütfen birkaç saniye sonra tekrar deneyin veya farklı kelimelerle arama yapın."
        print(f"Error: {e}")
    
    await thinking.remove()
    
    answer = cl.Message(content=content, author="Ayakkabı Asistanı 👟")
    
    messages.append({"role": "assistant", "content": content})
    cl.user_session.set("messages", messages)
    await answer.send()

@cl.set_starters
async def set_starts() -> List[cl.Starter]:
    return [
        cl.Starter(
            label="Gala Etkinliği İçin Şık Topuklu",
            message="Bu akşam bir gala etkinliğine katılacağım. Bordo bir elbise giyeceğim ve çok şık görünmek istiyorum. Hangi topuklu ayakkabı önerirsiniz?",
        ),
        cl.Starter(
            label="Yeni İş İçin Formal Ayakkabı",
            message="Önümüzdeki hafta yeni işime başlıyorum. Hem şık hem de tüm gün rahat olacağım formal bir ayakkabı önerebilir misiniz?",
        ),
        cl.Starter(
            label="Kış İçin Sıcak ve Stylish Bot",
            message="Kış geliyor ve hem şık görünmek hem de ayaklarımı sıcak tutmak istiyorum. Su geçirmez özellikli şık bot önerileriniz var mı?",
        ),
        cl.Starter(
            label="Spor ve Günlük Kombini",
            message="Hem spor yaparken hem de günlük hayatta giyebileceğim, tarzımı yansıtan ve ultra rahat bir ayakkabı arıyorum.",
        ),
        cl.Starter(
            label="Trend ve Benzersiz Bir Şeyler",
            message="Sıradanlıktan sıkıldım! 2025'in trend ayakkabı modellerinden, göz alıcı ve benzersiz bir şeyler önerir misiniz?",
        )
    ]

