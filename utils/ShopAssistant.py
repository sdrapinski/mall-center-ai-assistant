import json
import pandas as pd

class ShopAssistant:
    def __init__(self, dataframe, openai_client):
        """
        Inicjalizacja asystenta zakupowego.
        
        Args:
            dataframe (pd.DataFrame): Załadowany DataFrame ze sklepami.
            openai_client: Klient OpenAI.
        """
        self.df = dataframe
        self.client = openai_client
        self.all_categories = set()
        
        # 1. Przetworzenie DataFrame, aby wyciągnąć unikalne kategorie
        for categories_str in self.df['CATEGORIES']:
            if pd.notna(categories_str): # Sprawdzenie na NaN
                cats = [c.strip() for c in categories_str.split(',')]
                self.all_categories.update(cats)
        
        print(f"Asystent załadowany. Znaleziono {len(self.all_categories)} unikalnych kategorii.")

    def analyze_intent(self, question):
        """
        Analizuje tekst pytania i zwraca pasujące sklepy z DataFrame.
        """
        # 2. Definicja Function Calling z dynamiczną listą kategorii
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "filter_categories",
                    "description": "Wybiera odpowiednie kategorie zakupowe na podstawie zapytania użytkownika.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "matched_categories": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": list(self.all_categories) 
                                },
                                "description": "Lista kategorii pasujących do intencji użytkownika."
                            }
                        },
                        "required": ["matched_categories"]
                    }
                }
            }
        ]

        # 3. Zapytanie do OpenAI
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o", 
                messages=[
                    {"role": "system", "content": "Jesteś asystentem w centrum handlowym. Twoim zadaniem jest przypisanie luźnego zapytania klienta do konkretnych kategorii dostępnych w bazie."},
                    {"role": "user", "content": question}
                ],
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "filter_categories"}}
            )

            # 4. Parsowanie odpowiedzi
            tool_call = response.choices[0].message.tool_calls[0]
            arguments = json.loads(tool_call.function.arguments)
            found_categories = arguments.get("matched_categories", [])
            
            # 5. Filtrowanie DataFrame (szukamy sklepów, które mają te kategorie)
            matched_shops = []
            
            # Iterujemy po DF i sprawdzamy kategorie
            for index, row in self.df.iterrows():
                shop_cats = [c.strip() for c in row['CATEGORIES'].split(',')]
                # Sprawdzamy cześć wspólną zbiorów (czy sklep ma którąś z szukanych kategorii)
                if set(shop_cats).intersection(set(found_categories)):
                    matched_shops.append({"name" :row['NAME'], "id": row['ID']}) 

            return {
                "input_text": question,
                "detected_categories": found_categories,
                "matching_shops": matched_shops
            }

        except Exception as e:
            print(f"Błąd podczas analizy intencji: {e}")
            return None