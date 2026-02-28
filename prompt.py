"""
Reference:
 - Prompts are from [graphrag](https://github.com/microsoft/graphrag)
"""

GRAPH_FIELD_SEP = "<SEP>"
PROMPTS = {}
PROMPTS["entity_extraction"] = """你是法律文本结构化抽取专家。请从下面的中文法律文本中抽取“实体”，并严格使用规定的输出格式与分隔符。

要求：
- 仅抽取与法律意义相关的实体，优先：法律法规/司法解释/法条/法律概念/主体（国家机关/组织/个人）/程序/权利/义务/责任/措施/地域/时间/证据等。
- 每个实体给出精炼的中文描述（≤50字），避免复述整段原文。
- 对同义称呼用统一名称（如“民法典”“中华人民共和国民法典”统一为“中华人民共和国民法典”）。
- 严格使用分隔符，不要输出多余文本、编号、markdown或解释性内容。

实体类型限定在：[{entity_types}]

输出格式（每条一行，行与行用{record_delimiter}分隔，最后以{completion_delimiter}结束）：
("entity"{tuple_delimiter}"name"{tuple_delimiter}"type"{tuple_delimiter}"description")

示例：
("entity"{tuple_delimiter}"《中华人民共和国民法典'"{tuple_delimiter}"法律法规"{tuple_delimiter}"中国民事基本法，含总则/物权/合同等编"){record_delimiter}
("entity"{tuple_delimiter}"第一条（立法目的）"{tuple_delimiter}"法条"{tuple_delimiter}"阐明立法目的与依据"){record_delimiter}
("entity"{tuple_delimiter}"权利能力"{tuple_delimiter}"法律概念"{tuple_delimiter}"民事主体享有权利与承担义务的资格"){record_delimiter}{completion_delimiter}

待抽取文本：
{input_text}
"""
PROMPTS["entiti_continue_extraction"] = """继续在相同文本上抽取遗漏的“实体”，避免与之前已抽取的重复。严格沿用相同输出格式与分隔符：
("entity"{tuple_delimiter}"name"{tuple_delimiter}"type"{tuple_delimiter}"description")

仅输出新增条目，逐行给出，各行之间用{record_delimiter}分隔，最后以{completion_delimiter}结束。不要输出解释、无关内容或空行。"""

PROMPTS["entiti_if_loop_extraction"] = """请仅回答 yes 或 no：
如果你认为该文本中还有尚未抽取但应当抽取的实体，则回答 yes；否则回答 no。不要输出任何其它内容。"""

PROMPTS["relation_extraction"] = """你是法律知识图谱关系抽取专家。现给出一段中文法律文本与其中已识别的实体列表，请抽取“关系”。仅在给定实体之间判断关系，不要引入未在实体列表中的新实体。

给定实体列表（逗号分隔）：
{entities}

可用关系类型（仅限以下，选其一；必要时可在描述中补充）：
- 规定/定义
- 适用于/适用范围
- 引用/依据/参照
- 修改/废止/替代
- 解释/阐释
- 主体-权利
- 主体-义务
- 主体-责任
- 条件-结果
- 程序-步骤/要求
- 层级/隶属/归属
- 空间/地域范围
- 时间/生效/失效

输出格式（每条一行，行与行用{record_delimiter}分隔，最后以{completion_delimiter}结束）：
("relationship"{tuple_delimiter}"src"{tuple_delimiter}"tgt"{tuple_delimiter}"evidence"{tuple_delimiter}weight)

- src：源实体名，必须来自实体列表
- tgt：目标实体名，必须来自实体列表
- evidence：从原文中截取的最小充分中文短句（≤60字），可轻微改写但需可溯源
- weight：关系强度分数（0~10，建议7~9，保留1位小数或整数）

示例：
("relationship"{tuple_delimiter}"《中华人民共和国民法典’"{tuple_delimiter}"权利能力"{tuple_delimiter}"自然人自出生到死亡具有民事权利能力"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"监护人"{tuple_delimiter}"未成年人"{tuple_delimiter}"监护人应当按照最有利于被监护人的原则履行职责"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"失踪宣告"{tuple_delimiter}"自然人"{tuple_delimiter}"自然人下落不明满二年可以申请宣告失踪"{tuple_delimiter}8){record_delimiter}{completion_delimiter}

待抽取文本：
{input_text}
"""
PROMPTS[
    "community_report"
] = """You are an AI assistant that helps a human analyst to perform general information discovery. 
Information discovery is the process of identifying and assessing relevant information associated with certain entities (e.g., organizations and individuals) within a network.

# Goal
Write a comprehensive report of a community, given a list of entities that belong to the community as well as their relationships and optional associated claims. The report will be used to inform decision-makers about information associated with the community and their potential impact. The content of this report includes an overview of the community's key entities, their legal compliance, technical capabilities, reputation, and noteworthy claims.

# Report Structure

The report should include the following sections:

- TITLE: community's name that represents its key entities - title should be short but specific. When possible, include representative named entities in the title.
- SUMMARY: An executive summary of the community's overall structure, how its entities are related to each other, and significant information associated with its entities.
- IMPACT SEVERITY RATING: a float score between 0-10 that represents the severity of IMPACT posed by entities within the community.  IMPACT is the scored importance of a community.
- RATING EXPLANATION: Give a single sentence explanation of the IMPACT severity rating.
- DETAILED FINDINGS: A list of 5-10 key insights about the community. Each insight should have a short summary followed by multiple paragraphs of explanatory text grounded according to the grounding rules below. Be comprehensive.

Return output as a well-formed JSON-formatted string with the following format:
    {{
        "title": <report_title>,
        "summary": <executive_summary>,
        "rating": <impact_severity_rating>,
        "rating_explanation": <rating_explanation>,
        "findings": [
            {{
                "summary":<insight_1_summary>,
                "explanation": <insight_1_explanation>
            }},
            {{
                "summary":<insight_2_summary>,
                "explanation": <insight_2_explanation>
            }}
            ...
        ]
    }}

# Grounding Rules
Do not include information where the supporting evidence for it is not provided.
# Naming Rule (Important)
The report's TITLE must NOT be the same as or identical to any individual entity name.
If needed, combine two or more key elements or describe the function/purpose of the collective to create a distinct and informative title.


# Example Input
-----------
Text:
```
Entities:
```csv
id,entity,type,description
5,VERDANT OASIS PLAZA,geo,Verdant Oasis Plaza is the location of the Unity March
6,HARMONY ASSEMBLY,organization,Harmony Assembly is an organization that is holding a march at Verdant Oasis Plaza
```
Relationships:
```csv
id,source,target,description
37,VERDANT OASIS PLAZA,UNITY MARCH,Verdant Oasis Plaza is the location of the Unity March
38,VERDANT OASIS PLAZA,HARMONY ASSEMBLY,Harmony Assembly is holding a march at Verdant Oasis Plaza
39,VERDANT OASIS PLAZA,UNITY MARCH,The Unity March is taking place at Verdant Oasis Plaza
40,VERDANT OASIS PLAZA,TRIBUNE SPOTLIGHT,Tribune Spotlight is reporting on the Unity march taking place at Verdant Oasis Plaza
41,VERDANT OASIS PLAZA,BAILEY ASADI,Bailey Asadi is speaking at Verdant Oasis Plaza about the march
43,HARMONY ASSEMBLY,UNITY MARCH,Harmony Assembly is organizing the Unity March
```
```
Output:
{{
    "title": "Verdant Oasis Plaza and Unity March",
    "summary": "The community revolves around the Verdant Oasis Plaza, which is the location of the Unity March. The plaza has relationships with the Harmony Assembly, Unity March, and Tribune Spotlight, all of which are associated with the march event.",
    "rating": 5.0,
    "rating_explanation": "The impact severity rating is moderate due to the potential for unrest or conflict during the Unity March.",
    "findings": [
        {{
            "summary": "Verdant Oasis Plaza as the central location",
            "explanation": "Verdant Oasis Plaza is the central entity in this community, serving as the location for the Unity March. This plaza is the common link between all other entities, suggesting its significance in the community. The plaza's association with the march could potentially lead to issues such as public disorder or conflict, depending on the nature of the march and the reactions it provokes."
        }},
        {{
            "summary": "Harmony Assembly's role in the community",
            "explanation": "Harmony Assembly is another key entity in this community, being the organizer of the march at Verdant Oasis Plaza. The nature of Harmony Assembly and its march could be a potential source of threat, depending on their objectives and the reactions they provoke. The relationship between Harmony Assembly and the plaza is crucial in understanding the dynamics of this community."
        }},
        {{
            "summary": "Unity March as a significant event",
            "explanation": "The Unity March is a significant event taking place at Verdant Oasis Plaza. This event is a key factor in the community's dynamics and could be a potential source of threat, depending on the nature of the march and the reactions it provokes. The relationship between the march and the plaza is crucial in understanding the dynamics of this community."
        }},
        {{
            "summary": "Role of Tribune Spotlight",
            "explanation": "Tribune Spotlight is reporting on the Unity March taking place in Verdant Oasis Plaza. This suggests that the event has attracted media attention, which could amplify its impact on the community. The role of Tribune Spotlight could be significant in shaping public perception of the event and the entities involved."
        }}
    ]
}}


# Real Data

Use the following text for your answer. Do not make anything up in your answer.

Text:
```
{input_text}
```

The report should include the following sections:

- TITLE: community's name that represents its key entities - title should be short but specific. When possible, include representative named entities in the title.
- SUMMARY: An executive summary of the community's overall structure, how its entities are related to each other, and significant information associated with its entities.
- IMPACT SEVERITY RATING: a float score between 0-10 that represents the severity of IMPACT posed by entities within the community.  IMPACT is the scored importance of a community.
- RATING EXPLANATION: Give a single sentence explanation of the IMPACT severity rating.
- DETAILED FINDINGS: A list of 5-10 key insights about the community. Each insight should have a short summary followed by multiple paragraphs of explanatory text grounded according to the grounding rules below. Be comprehensive.

Return output as a well-formed JSON-formatted string with the following format:
    {{
        "title": <report_title>,
        "summary": <executive_summary>,
        "rating": <impact_severity_rating>,
        "rating_explanation": <rating_explanation>,
        "findings": [
            {{
                "summary":<insight_1_summary>,
                "explanation": <insight_1_explanation>
            }},
            {{
                "summary":<insight_2_summary>,
                "explanation": <insight_2_explanation>
            }}
            ...
        ]
    }}

# Grounding Rules
Do not include information where the supporting evidence for it is not provided.

Output:
"""





PROMPTS["summary_clusters_new"] = """
You are tasked with analyzing a set of entity descriptions and a given list of meta attributes. Your goal is to extract at least one *attribute entity* from the entity descriptions. The extracted entity must match the type of at least one meta attribute from the provided list, and it must be directly relevant to the described entities. The relationship between the entity and the original entities must be logical and clearly identifiable from the text.

❗️Your output MUST strictly follow the **output format and syntax** described below. Do NOT include any explanation, headings, or extra information. Only return raw structured outputs.

---

�� Output Format (REQUIRED):
Each output must be in one of the two formats below (do not change any part of the structure):

1. For each identified attribute entity (must match meta_attribute_list):

("entity"{tuple_delimiter}"<entity_name>"{tuple_delimiter}"<entity_type>"{tuple_delimiter}"<entity_description>"){record_delimiter}

2. For each valid relationship between a described entity and an attribute entity:

("relationship"{tuple_delimiter}"<source_entity>"{tuple_delimiter}"<target_entity>"{tuple_delimiter}"<relationship_description>"{tuple_delimiter}<relationship_strength>"){record_delimiter}

Finally, end the output with this token exactly:
{completion_delimiter}

�� Self Check Before Submitting:
- All entities and relationships are enclosed in parentheses and use double quotes.
- Fields use {tuple_delimiter} to separate.
- Each record ends with {record_delimiter}.
- Output ends with {completion_delimiter}.
- No explanations, titles, markdown, or extra text outside required formats.

---

�� Example:
Input:
Meta attribute list: ["company", "location"]
Entity description list: [("Instagram", "Instagram is a software developed by Meta..."), ("Facebook", "Facebook is a social networking platform..."), ("WhatsApp", "WhatsApp Messenger: A messaging app of Meta...")]

Output:
("entity"{tuple_delimiter}"Meta"{tuple_delimiter}"company"{tuple_delimiter}"Meta, formerly known as Facebook, Inc., is an American multinational technology conglomerate. It is known for its various online social media services."){record_delimiter}
("relationship"{tuple_delimiter}"Instagram"{tuple_delimiter}"Meta"{tuple_delimiter}"Instagram is a software developed by Meta."{tuple_delimiter}8.5){record_delimiter}
("relationship"{tuple_delimiter}"Facebook"{tuple_delimiter}"Meta"{tuple_delimiter}"Facebook is owned by Meta."{tuple_delimiter}9.0){record_delimiter}
("relationship"{tuple_delimiter}"WhatsApp"{tuple_delimiter}"Meta"{tuple_delimiter}"WhatsApp Messenger is a messaging app of Meta."{tuple_delimiter}8.0){record_delimiter}
{completion_delimiter}

---

�� Task:
Input:
Meta attribute list: {meta_attribute_list}
Entity description list: {entity_description_list}

#######
Output:
"""

# TYPE的定义
PROMPTS["DEFAULT_ENTITY_TYPES"] = ["organization", "person", "geo", "event"]
PROMPTS["META_ENTITY_TYPES"] = [
    "法律法规", "司法解释", "法条", "法律概念", "主体-国家机关",
    "主体-组织机构", "主体-公民个人", "程序/制度", "权利", "义务",
    "责任/制裁", "措施/手段", "地域/机构层级", "时间/生效信息", "证据/文件"
]
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "||"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "###END###"

PROMPTS[
    "local_rag_response"
] = """---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, 
summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}


---Data tables---

{context_data}



Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""





PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["default_text_separator"] = [
    # Paragraph separators
    "\n\n",
    "\r\n\r\n",
    # Line breaks
    "\n",
    "\r\n",
    # Sentence ending punctuation
    "。",  # Chinese period
    "．",  # Full-width dot
    ".",  # English period
    "！",  # Chinese exclamation mark
    "!",  # English exclamation mark
    "？",  # Chinese question mark
    "?",  # English question mark
    # Whitespace characters
    " ",  # Space
    "\t",  # Tab
    "\u3000",  # Full-width space
    # Special characters
    "\u200b",  # Zero-width space (used in some Asian languages)
]

PROMPTS["cluster_cluster_relation_old"]="""
You are given the descriptions of two communities and their relationships between nodes. Based on these descriptions, summarize the relationship between the two communities in no more than 50 words. Focus on how the communities interact, collaborate, or contribute to each other.

Format:

Input:

Community A Name: {community_a}

Community A Description: {community_a_description}

Community B Name: {community_b}

Community B Description: {community_b}

Relations Between Them: {relation_infromation}

Output:
A concise summary (≤50 words) explaining the relationship between Community A and Community B, including the collaboration or roles each community plays in relation to the other.

Example:

Input:

Community A Name: WTO and Its Core Divisions

Community A Description: 'An economist and author of the World Trade Report 2020, contributing to economic research.'

Community B Name: WTO World Trade Report 2020 Contributors

Community B Description: 'The community consists of key figures from the World Trade Organization (WTO) who played significant roles in preparing and contributing to the World Trade Report 2020.'

Relations Between Them:

'relationship<|>World Trade Report 2020<|>Xiaozhun Yi<|>Indicates that the report was prepared under the general responsibility of Xiaozhun Yi.'

'relationship<|>World Trade Report 2020<|>Ankai Xu<|>Indicates that Ankai Xu was responsible for coordinating the World Trade Report 2020.'

'relationship<|>World Trade Report 2020<|>Robert Koopman<|>Indicates that the World Trade Report 2020 was prepared under Robert Koopman’s general responsibility.'

[Additional relationships...]

Output:
The WTO and Its Core Divisions community provides the institutional and research backbone for the WTO World Trade Report 2020 Contributors community, whose members—many from WTO divisions—collaboratively authored the report. Their relationship reflects a functional collaboration between core WTO units and designated contributors to produce key economic analysis.
"""
PROMPTS["summary_entities_old"]="""
You are an assistant that condenses multiple descriptions of a given entity into a single, concise summary. The final output should preserve all core information, use natural and accurate language, and stay within 100 tokens limit.
Format:

Input:
Entity Name: {entity_name} 
Entity Descriptions: {description}

Output:
Concise summary capturing all essential information, within 50 tokens

Example:

Input:
Entity Name: World Trade Report  
Entity Descriptions: A document published by the WTO that analyzes global trade trends and issues. | A comprehensive document analyzing global trade trends, impacts of policies, and economic developments. | An annual publication by the WTO analyzing global trade trends and issues, focusing on specific themes each year.  

Output:
An annual WTO publication analyzing global trade, policy impacts, and economic trends, with a yearly thematic focus.


"""
PROMPTS["summary_entities"]="""
# Role: Concise Summary Assistant

## Profile
- author: LangGPT
- version: 1.0
- language: English
- description: You are a summarization assistant tasked with condensing multiple descriptions of a given entity into a single, natural, and accurate summary.

## Skills
- Identify and preserve core information across inputs
- Perform effective linguistic compression without losing key content
- Use fluent, professional, and contextually appropriate language
- Maintain summary length under strict token limits

## Goals
- Combine all essential details into one concise summary
- Ensure the output is no longer than 100 tokens
- Maintain completeness and fluency of expression

## OutputFormat
Format:
Input:
Entity Name: {entity_name}
Entity Descriptions: {description}

Output:

<Concise summary capturing all essential information, under 100 tokens>

## Rules
- Do not omit any core fact that appears in the original descriptions
- Rephrase or combine sentences for clarity and brevity
- Keep the tone objective and informative
- Do not introduce any new or speculative content
- Ensure the final summary is grammatically correct and stylistically natural

## Example
Input:
Entity Name: World Trade Report
Entity Descriptions: A document published by the WTO that analyzes global trade trends and issues. | A comprehensive document analyzing global trade trends, impacts of policies, and economic developments. | An annual publication by the WTO analyzing global trade trends and issues, focusing on specific themes each year.

Output:
An annual WTO publication analyzing global trade, policy impacts, and economic trends, with a yearly thematic focus.


"""
PROMPTS["summary_entities_zh"]="""
# 角色: 简洁摘要助手

## 简介
- 作者: LangGPT
- 版本: 1.0
- 语言: 中文
- 描述: 你是一个摘要助手，负责将给定实体的多个描述合并为一个简洁、自然且准确的中文摘要。

## 技能
- 识别并保留输入中的核心信息
- 在不丢失关键内容的前提下进行有效的语言压缩
- 使用流畅、专业且符合语境的中文表达
- 严格控制摘要长度在token限制内

## 目标
- 将所有重要细节合并为一个简洁摘要
- 确保输出不超过100个token
- 保持完整性和表达的流畅性

## 输出格式
格式:
输入:
实体名称: {entity_name}
实体描述: {description}

输出:

<简洁摘要，包含所有重要信息，不超过100个token>

## 规则
- 不要省略原始描述中出现的任何核心事实
- 重新措辞或合并句子以提高清晰度和简洁性
- 保持客观和信息的语调
- 不要引入任何新的或推测性的内容
- 确保最终摘要语法正确且风格自然
- 使用中文输出

## 示例
输入:
实体名称: 《中华人民共和国民法典》
实体描述: 中国民事基本法，含总则/物权/合同等编 | 规范自然人、法人和非法人组织之间民事关系的法律 | 2021年1月1日起施行的民事法律规范

输出:
《中华人民共和国民法典》是2021年1月1日起施行的中国民事基本法，规范自然人、法人和非法人组织之间的民事关系，包含总则、物权、合同等各编。

"""
PROMPTS[
    "aggregate_entities"
]="""
# Role: Entity Aggregation Analyst

## Profile
- author: LangGPT
- version: 1.0
- language: English
- description: You are an expert in concept synthesis. Your task is to identify a meaningful aggregate entity from a set of related entities and extract structured insights based solely on provided evidence.

## Skills
- Abstraction and naming of collective concepts based on entity types
- Structured summarization and typology recognition
- Comparative analysis across multiple entities
- Strict grounding to provided data (no hallucinated content)

## Goals
- Derive a meaningful aggregate entity that broadly represents the given entity set
- The aggregate entity name must not match any single entity in the set
- Provide an accurate and concise description of the aggregate entity reflecting shared characteristics
- Extract 5–10 structured findings about the entity set based on grounded evidence

## OutputFormat
Format:
Input: 
{input_text}

Output: 
{{
      "entity_name": "<name>",
      "entity_description": "<brief description summarizing the shared traits and structure>",
      "findings": [
        {{
          "summary": "<summary_1>",
          "explanation": "<explanation_1>"
        }},
        {{
          "summary": "<summary_2>",
          "explanation": "<explanation_2>"
        }}
        // ...
      ]
    }}

## Rules
- Grounding Rule: All content must be based solely on the provided entity set — no external assumptions
- Naming Rule: The aggregate entity name must not be identical to any single entity; it should reflect a composite structure, function, or theme
- Each finding must include a concise summary and a detailed explanation
- Avoid adding speculative or unsupported interpretations

## Workflows
1. Review the list of entities, focusing on types, descriptions, and relational structure
2. Synthesize a generalized name that best represents the full entity set
3. Write a clear, evidence-based description of the aggregate entity
4. Extract and elaborate on key findings, emphasizing structure, purpose, and interconnections

"""
PROMPTS["aggregate_entities_zh"]="""
# 角色: 法律实体聚合分析专家

## 简介
- 作者: LangGPT
- 版本: 2.0
- 语言: 中文
- 描述: 你是法律概念综合专家。你的任务是从一组相关的法律实体中识别一个有意义的聚合实体，并基于提供的证据提取结构化洞察。

## 技能
- 基于实体类型的集体概念抽象和命名
- 结构化摘要和类型识别
- 跨多个实体的比较分析
- 严格基于提供的数据（不编造内容）
- 理解法律术语和法律体系结构

## 目标
- 推导出一个有意义的聚合实体，广泛代表给定的实体集合
- 聚合实体名称不能与集合中任何单个实体名称相同
- 提供准确简洁的聚合实体描述，反映共享特征
- 基于有根据的证据提取5-10个结构化发现

## 输出格式
格式:
输入: 
{input_text}

输出: 
{{
      "entity_name": "<聚合实体名称>",
      "entity_description": "<简要描述，总结共享特征和结构>",
      "findings": [
        {{
          "summary": "<摘要1>",
          "explanation": "<解释1>"
        }},
        {{
          "summary": "<摘要2>",
          "explanation": "<解释2>"
        }}
        // ... 5-10个发现
      ]
    }}

## 规则
- 基础规则: 所有内容必须仅基于提供的实体集合，不做外部假设
- 命名规则: 
  - 聚合实体名称不能与任何单个实体名称相同
  - 应反映复合结构、功能或主题
  - 使用规范的中文法律术语
  - 名称应简洁、准确、具有概括性
- 描述规则:
  - 描述应总结实体的共同特征和法律意义
  - 突出法律体系中的位置和作用
  - 使用专业但易懂的中文法律语言
- 发现规则:
  - 每个发现必须包括简洁的摘要和详细的解释
  - 发现应基于实体类型、描述和关系结构
  - 避免添加推测性或不受支持的解读
  - 重点关注法律概念、规范结构、权利义务关系等

## 法律知识图谱特殊要求
- 识别法律体系层次: 区分法律法规、法条、法律概念、主体等不同类型
- 理解法律关系: 识别规定、适用、引用、解释等法律关系类型
- 保持法律准确性: 使用准确的法律术语，避免模糊表达
- 体现法律逻辑: 发现应体现法律规范的逻辑结构和体系性

## 工作流程
1. 审查实体列表，重点关注类型、描述和关系结构
2. 识别实体的法律类型（法律法规/法条/法律概念/主体/程序/权利/义务等）
3. 综合一个最能代表完整实体集合的通用名称
4. 编写清晰、基于证据的聚合实体描述
5. 提取并阐述关键发现，强调结构、目的和相互联系
6. 验证所有内容都基于提供的证据，没有编造

## 示例
假设输入包含以下实体:
- 《中华人民共和国民法典》
- 第一条（立法目的）
- 权利能力
- 行为能力

可能的输出:
{{
  "entity_name": "民事基本法律制度",
  "entity_description": "涵盖民事法律基本规范、主体资格和权利能力的法律制度体系，包括民法典核心条款和基础法律概念",
  "findings": [
    {{
      "summary": "以民法典为核心的法律规范体系",
      "explanation": "该聚合以《中华人民共和国民法典》为核心，包含立法目的条款和基础法律概念，形成完整的民事基本法律制度框架"
    }},
    {{
      "summary": "主体资格与权利能力并重",
      "explanation": "聚合中包含权利能力和行为能力等核心概念，体现了对民事主体资格和权利能力的重视"
    }}
  ]
}}

"""
PROMPTS["cluster_cluster_relation"]="""
# Role: Inter-Aggregation Relationship Analyst

## Profile
- author: LangGPT
- version: 1.1
- language: English
- description: You specialize in analyzing relationships between two aggregation entities. Your goal is to synthesize one high-level, abstract summary sentence describing how two named aggregations are connected, based solely on their descriptions and sub-entity relationships.

## Skills
- Aggregated reasoning across entity groups
- Abstraction of cross-entity relationships
- Formal summarization under strict constraints
- Strong grounding without repetition or speculation

## Goals
- Produce a single-sentence summary (≤{tokens} words) explaining the nature of the relationship between two aggregation entities
- Avoid reproducing individual sub-entity relationships
- Emphasize structural, functional, or thematic connections at the group level

---

## InputFormat
Aggregation A Name: {entity_a}
Aggregation A Description: {entity_a_description}

Aggregation B Name: {entity_b}
Aggregation B Description: {entity_b_description}

Sub-Entity Relationships:
{relation_information}
---

## OutputFormat
<Single-sentence explanation (≤{tokens} words) summarizing the relationship between Aggregation A and Aggregation B. Use abstract group-level language and do not include names or specific node-level relationships.>

---

## Rules

- DO NOT output `relationship<|>` lines or copy sub-entity relationship descriptions
- DO NOT name specific sub-entities (e.g., individuals)
- DO NOT use the term "community"; always refer to "aggregation," "group," "collection," or thematic equivalents
- DO use collective terms (e.g., "external reviewers," "trade policy actors")
- The sentence must be ≤{tokens} words, factual, grounded, and in formal English
- The relationship must reflect an **aggregation-level abstraction**, such as:
  - support/collaboration
  - review/feedback
  - functional alignment
  - domain linkage (e.g., one produces work, the other evaluates it)

---

## Example

### Input:
Aggregation A Name: WTO External Contributors  
Aggregation A Description: A group of economists and trade policy experts who provided feedback on early drafts of WTO reports.  

Aggregation B Name: WTO Flagship Reports  
Aggregation B Description: Core analytical publications from the WTO addressing international trade issues.  

Sub-Entity Relationships:
- Person A → early drafts of WTO report → gave feedback  
- Person B → early drafts → reviewed document  
...

### ✅ Output:
WTO External Contributors played an advisory role to the WTO Flagship Reports aggregation by offering critical expert feedback on preliminary drafts, strengthening the analytical rigor and credibility of the final publications.

"""
PROMPTS["cluster_cluster_relation_zh"]="""
# 角色: 法律聚合实体关系分析专家

## 简介
- 作者: LangGPT
- 版本: 2.0
- 语言: 中文
- 描述: 你专门分析两个聚合实体之间的关系。你的目标是基于它们的描述和子实体关系，综合一个高级的、抽象的摘要句子，描述两个命名的聚合实体是如何连接的。

## 技能
- 跨实体组的聚合推理
- 跨实体关系的抽象
- 严格约束下的正式摘要
- 强基础性，不重复或推测
- 理解法律关系的类型和特征

## 目标
- 生成一个单句摘要（≤{tokens}字），解释两个聚合实体之间关系的性质
- 避免重复单个子实体关系
- 强调群体层面的结构、功能或主题连接

---

## 输入格式
聚合实体A名称: {entity_a}
聚合实体A描述: {entity_a_description}

聚合实体B名称: {entity_b}
聚合实体B描述: {entity_b_description}

子实体关系:
{relation_information}
---

## 输出格式
<单句解释（≤{tokens}字），总结聚合实体A和聚合实体B之间的关系。使用抽象的群体层面语言，不包含名称或特定的节点级别关系。>

---

## 规则

- 不要输出 `relationship<|>` 行或复制子实体关系描述
- 不要命名特定的子实体（例如，具体的法条名称、具体的主体名称）
- 不要使用"社区"一词；始终使用"聚合"、"群体"、"集合"或主题等价词
- 使用集合术语（例如，"法律规范制定者"、"权利主体"、"程序执行机构"）
- 句子必须 ≤{tokens}字，事实性，有根据，使用正式的中文
- 关系必须反映**聚合层面的抽象**，例如：
  - 规定/定义关系（一个聚合规定另一个聚合的内容）
  - 适用/适用范围关系（一个聚合适用于另一个聚合）
  - 引用/依据关系（一个聚合引用或依据另一个聚合）
  - 解释/阐释关系（一个聚合解释另一个聚合）
  - 修改/废止/替代关系（一个聚合修改、废止或替代另一个聚合）
  - 功能对齐（两个聚合在法律体系中具有互补功能）
  - 领域链接（例如，一个产生规范，另一个执行规范）

## 法律知识图谱特殊要求
- 识别法律关系类型: 区分规定、适用、引用、解释、修改等法律关系
- 使用法律术语: 使用准确的中文法律术语描述关系
- 体现法律逻辑: 关系描述应体现法律规范的逻辑结构和体系性
- 保持抽象性: 在聚合层面描述关系，不涉及具体法条或具体主体

---

## 示例

### 输入:
聚合实体A名称: 民事主体法律制度
聚合实体A描述: 规范自然人、法人和非法人组织等民事主体的资格、权利能力和行为能力的法律规范集合

聚合实体B名称: 民事权利与义务规范
聚合实体B描述: 规定民事主体享有的权利和承担的义务的法律规范集合

子实体关系:
relationship<|>自然人<|>民事权利<|>自然人享有民事权利
relationship<|>法人<|>民事义务<|>法人承担民事义务
...

### ✅ 输出:
民事主体法律制度与民事权利与义务规范之间存在主体-权利映射关系，前者定义了参与民事活动的各类主体及其资格，后者规定了这些主体享有的权利和承担的义务，两者共同构成民事法律关系的完整框架。

"""
PROMPTS["response"]="""
# Role: Structured Data Response Generator

## Profile
- author: LangGPT
- version: 1.0
- language: English
- description: You are a precise summarization and reasoning assistant. Your task is to answer a user’s question based on tabular data inputs by generating a structured response that respects length and format constraints, using only grounded and verifiable information.

## Skills
- Data interpretation from structured input tables
- Factual summarization without extrapolation
- Response length control and format compliance
- Integration of relevant general knowledge only when supported

## Goals
- Generate a response that answers the user’s question based solely on provided data
- Conform to the expected response length and format
- Include only information with direct or clearly inferable support
- Omit or explicitly acknowledge any unsupported or unknown content

## Input
{context_data}

Output:
<Structured response that satisfies the question using only supported and summarized information from the input tables. Incorporates general knowledge only if directly supported by the data. If unknown, respond clearly with “I don’t know based on the available data.”>

## Rules
- Do not fabricate or speculate
- Do not include information without supporting evidence
- Use natural, accurate language within the target format
- If the answer is not evident in the data, clearly state “I don’t know based on the available data.”

## Workflows
1. Parse the user question and understand the required response format and length
2. Analyze and summarize the input tables to extract relevant, factual information
3. Synthesize a concise, format-matching response based solely on grounded data
4. Validate that each statement is traceable to the input or clearly marked as unknown


"""
PROMPTS[
    "rag_response"
] = """---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

Multiple Paragraphs


---Data tables---

{context_data}


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.

If you don't know the answer, just say so. Do not make anything up.

Do not include information where the supporting evidence for it is not provided.


Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""
PROMPTS["rag_response_zh"] = """---角色---

你是一个专业的法律知识问答助手，基于提供的法律知识图谱数据回答用户的问题。

---目标---

根据输入数据表中的信息，生成符合目标长度和格式的回答，总结所有相关信息，并适当结合相关的法律知识。
如果你不知道答案，请明确说明。不要编造信息。
不要包含没有支持证据的信息。

---目标回答长度和格式---

多段落格式

---数据表---

{context_data}

数据表包含以下部分：
1. entity_information: 检索到的实体信息（实体名称、父节点、描述）
2. aggregation_entity_information: 聚合实体信息（社区/群组层面的信息）
3. reasoning_path_information: 推理路径信息（实体之间的关系）
4. text_units: 原始文本块（实体来源的完整原始文本）

---目标---

根据输入数据表中的信息，生成符合目标长度和格式的回答，总结所有相关信息，并适当结合相关的法律知识。

如果你不知道答案，请明确说明。不要编造信息。

不要包含没有支持证据的信息。

---法律知识图谱特殊要求---

1. 准确性优先: 基于提供的实体信息和原始文本块回答问题，确保法律术语的准确性
2. 可追溯性: 回答应能追溯到提供的实体和文本块
3. 法律逻辑: 回答应体现法律规范的逻辑结构和体系性
4. 专业表达: 使用规范的中文法律术语，保持专业但易懂
5. 结构化回答: 使用适当的章节和注释，使用 markdown 格式

---回答格式建议---

1. 开头: 直接回答问题的核心要点
2. 主体: 基于实体信息和原始文本块详细阐述
3. 依据: 说明回答的法律依据（引用相关实体或文本块）
4. 总结: 简要总结关键点

---示例回答结构---

## 回答

[基于实体信息和原始文本块的核心回答]

### 详细说明

[基于检索到的实体和关系的详细解释]

### 法律依据

[引用相关的实体、关系或原始文本块]

### 总结

[关键要点的简要总结]
"""
PROMPTS["rag_response_article_zh"] = """---角色---

你是一个专业的法律知识问答助手，专门回答关于法条具体内容的问题。

---目标---

当用户询问法条的具体内容时，你需要：
1. 从 text_units（原始文本块）中找到包含该法条的完整文本
2. 直接引用法条的完整原文内容
3. 如果 text_units 中包含该法条，必须完整引用原文
4. 如果 text_units 中没有该法条，基于实体描述尽可能详细地说明

---数据表---

{context_data}

数据表包含以下部分：
1. entity_information: 检索到的实体信息（实体名称、父节点、描述）
   - 如果实体类型是"法条"，描述字段可能只包含简要说明，完整内容在 text_units 中
2. aggregation_entity_information: 聚合实体信息（社区/群组层面的信息）
3. reasoning_path_information: 推理路径信息（实体之间的关系）
4. text_units: 原始文本块（实体来源的完整原始文本）
   - **这是法条完整内容的关键来源**
   - 如果 text_units 中包含法条内容，必须完整引用

---法条查询特殊要求---

1. **只返回用户询问的法条**:
   - **重要**：用户询问的是特定法条（如"民法典第一条"），你只能返回该法条的内容
   - **不要**返回 text_units 中的其他法条内容（即使它们也在 text_units 中）
   - **不要**返回其他法律的法条（如用户问"民法典第一条"，不要返回"国家赔偿法"、"集会游行示威法"等其他法律的法条）
   - 如果 text_units 中包含多个法条，只提取并返回用户询问的那个法条

2. **优先使用原始文本块**: 
   - 如果 text_units 中包含用户询问的法条，必须完整引用原文
   - 不要只给出摘要或解释，要给出完整的法条原文

3. **法条格式要求**:
   - 保持法条的原始格式和标点
   - 如果法条有编号（如"第一条"、"第二十三条"），必须包含
   - 如果法条有标题（如"（立法目的）"），必须包含

4. **完整性要求**:
   - 如果法条内容较长，包含多个段落，必须全部引用
   - 不要省略法条的任何部分

5. **准确性要求**:
   - 法条内容必须与 text_units 中的原文完全一致
   - 不要修改、解释或补充法条内容
   - 如果需要解释，在引用原文后单独说明

---回答格式（法条查询）---

**重要**：只生成一次回答，不要重复生成。回答格式如下：

## 法条内容

[完整引用 text_units 中包含的法条原文，保持原始格式]

### 法条说明

[如果需要，可以基于实体描述和关系信息对法条进行说明]

### 法律依据

[说明该法条所属的法律法规]

**注意**：回答完成后立即停止，不要重复生成相同的内容。

---示例---

如果用户问："民法典第一条的内容是什么？"

如果 text_units 中包含：
"第一条 为了保护民事主体的合法权益，调整民事关系，维护社会和经济秩序，适应中国特色社会主义发展要求，弘扬社会主义核心价值观，根据宪法，制定本法。"

回答应该是：
## 法条内容

**第一条** 为了保护民事主体的合法权益，调整民事关系，维护社会和经济秩序，适应中国特色社会主义发展要求，弘扬社会主义核心价值观，根据宪法，制定本法。

### 法条说明

这是《中华人民共和国民法典》的立法目的条款，明确了民法典的制定目的和依据。

### 法律依据

本条款出自《中华人民共和国民法典》总则编。

---
"""