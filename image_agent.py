import os

def load_env_file_from_text(file_path):
    """
    Load environment variables from a text file where each line is KEY=VALUE.
    Lines starting with '#' are treated as comments.
    """
    if not os.path.isfile(file_path):
        print(f"⚠️  File not found: {file_path}")
        return

    print(f"✅ Loading environment variables from: {file_path}")
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                if "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()
                    print(f"🔑 Loaded {key.strip()}")
                else:
                    print(f"⚠️  Skipped invalid line: {line}")

# Usage
env_file_path = "api.txt"
load_env_file_from_text(env_file_path)

# Access the loaded variables
openai_api_key = os.getenv("OPENAI_API_KEY")
stability_api_key = os.getenv("STABILITY_API_KEY")

print(f"OPENAI_API_KEY: {openai_api_key[:5]}...") if openai_api_key else print("❌ OPENAI_API_KEY not set.")
print(f"STABILITY_API_KEY: {stability_api_key[:5]}...") if stability_api_key else print("❌ STABILITY_API_KEY not set.")

import img2img_tool as img2img
import inpainting_tool as inp
import text2image_tool as txt2img
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, initialize_agent
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents.openai_functions_agent.base import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain_core.tools import Tool,StructuredTool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import requests
import re
import random

# Initialize
llm = ChatOpenAI(model="gpt-4o", temperature=0.3, openai_api_key=openai_api_key)
#short_term_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#session_state = {"last_image_url": None, "last_prompt": None,"last_pattern":None,"last_color":None,"last_request":None,"last_part":None}


# ------------------ Prompts & Runnables ------------------
def create_chain(prompt_template):
    return prompt_template | llm

# ------------------- Prompt Examples -------------------
example_chain = create_chain(PromptTemplate.from_template(
    """
Give 3 concise creative car wrap design examples.
Each should include a pattern and color, no more than 10 words.

Examples:
1. Flames in red
2. Geometric lines in silver
3. Solid matte black color change
"""
))

# ------------------- Planning & Parsing -------------------
extract_design_chain = PromptTemplate.from_template(
    """
You are an expert car wrap creative assistant.

Based on the user's input, always extract the following **clearly and completely**, even if the user is vague or only expresses a mood or emotion.

Rules:
-  **Special color-only change rule (high priority)**:
    - If the user input is **only about changing the color** (like 'change color to red', 'make it matte black', 'I want it chrome gold') and does not mention pattern, object, mood, or parts:
        - Extract as:
            - Adjustment: Change the color to X.
            - Object_name: use last
            - Pattern: solid color
            - Color: X (as provided)
            - Style: photographic
            - Request: pure color, clean, no patterns
    - This rule takes priority when the user input clearly only focuses on color.
- If the user's input is **vague, only a mood, or only expresses emotions like 'make it cool', 'sleek', 'pop more', 'too boring'**, you MUST **infer the most fitting pattern, color, style, and request using your creative design knowledge.**
- You MUST always output **fully populated fields for Pattern, Color, Style, and Request.**
- You must NEVER return 'unknown', 'none', 'default', or any empty fields.
- When color or pattern is not mentioned, propose **the most likely or trendy matching color and pattern**.
- Always select **only ONE style** strictly from this list:
  enhance, anime, photographic, digital-art, comic-book, fantasy-art, line-art, analog-film, neon-punk, isometric, low-poly, origami, modeling-compound, cinematic, 3d-model, pixel-art, tile-texture.
- NEVER invent styles outside the list.
- Always summarize the user's **design goal, mood, or style intent into 'Request'.**
- If the user's input is short or only says 'add a dog', assume it's a design-level change unless explicitly specifying a car part.
- When the user says 'add a dog' → assume it's for the overall design unless they say part (e.g., hood). Extract as if creating a **main design with a dog pattern**.
- When the user only says 'change to red' → treat as design-level and infer **Pattern** (e.g., abstract swirls), **Style** (e.g., digital-art), and **Request** (e.g., bold, striking).
- You MUST always generate a full new design intent when the detected intent is 'initial', regardless of the user only mentioning adding an element or placing it somewhere (e.g., 'add a dog on the door').
- When intent is 'initial', ignore the casual phrasing and instead infer the most matching pattern, color, style, and request for a new car wrap.
- If the user only mentions an object like 'dog' or a part like 'door', infer the most logical matching pattern, color, style, and create a holistic design description.


Design principle examples:
- For corporate, formal, elegant: pattern 'geometric lines' or 'minimal lines', color 'dark blue', 'silver', style 'photographic'.
- For fun, cute: pattern 'dog and cat illustration', color 'natural fur colors', style 'anime'.
- For futuristic, tech: pattern 'robot illustration' or 'digital circuit', color 'silver and blue', style 'digital-art'.

Few-shot examples:

Input: "Add a dog on the door"
Output:
- Pattern: dog illustration
- Color: natural fur colors
- Style: digital-art
- Request: friendly, playful, modern

Input: "Change color to red"
Output:
- Pattern: solid color
- Color: red
- Style: tile-texture
- Request: pure color, clean, no patterns

Input: "Make it green"
Output:
- Pattern: solid color
- Color: green
- Style: photographic
- Request: pure color, clean, no patterns

Input: "I like red flames"
Output:
- Pattern: flames
- Color: red
- Style: digital-art
- Request: aggressive, high-energy

Input: "geometric lines and blue"
Output:
- Pattern: geometric lines
- Color: blue
- Style: digital-art
- Request: sleek, modern

Input: "dog and cat"
Output:
- Pattern: dog and cat illustration
- Color: natural fur colors
- Style: anime
- Request: cute, friendly

Input: "sleek luxury look in silver"
Output:
- Pattern: minimal lines
- Color: silver
- Style: photographic
- Request: premium, classy

Input: "Corporate business theme with dark blue"
Output:
- Pattern: business
- Color: dark blue
- Style: photographic
- Request: professional, serious

Input: "add a robot"
Output:
- Pattern: robot illustration
- Color: metallic silver and blue
- Style: digital-art
- Request: futuristic, tech-inspired

Input: "pop more"
Output:
- Pattern: abstract swirls
- Color: neon colors
- Style: neon-punk
- Request: vibrant, eye-catching

Input: "too boring, make it bold"
Output:
- Pattern: bold geometric shapes
- Color: bright red and black
- Style: digital-art
- Request: bold, aggressive

Input: "Add a dog"
Output:
- Pattern: dog illustration
- Color: natural fur colors
- Style: digital-art
- Request: playful, friendly

Input: "Change to red"
Output:
- Pattern: abstract waves
- Color: red
- Style: digital-art
- Request: bold, passionate

Input: "sleek look"
Output:
- Pattern: minimal lines
- Color: silver
- Style: photographic
- Request: sleek, premium

Input: {input}

Output:
- Pattern:
- Color:
- Style:
- Request:
"""
) | llm





extract_adjustment_chain = PromptTemplate.from_template(
    """
You are an expert creative assistant. Based on the user's input, extract the adjustment details **strictly in JSON format**.

--- Definitions ---
- This is a **global adjustment**, not for specific car parts.
- NEVER extract 'part'. It must always be empty or not included.
- If the user explicitly mentions car parts (hood, door, etc.), this is wrong usage → the reflection agent will handle it.
- 'Object_name' refers to the design element being changed or added (e.g., stars, leaves, flames, dragon).

--- Extraction Rules ---
- **Special color-only change rule (high priority)**:
    - If the user input is **only about changing the color** (e.g., 'change color to red', 'make it matte black', 'I want it chrome gold') and does not mention pattern, object, mood, or parts:
        - Extract as:
            - "adjustment": "Change the color to X."
            - "object_name": "use last"
            - "pattern": "solid color"
            - "color": "X"
            - "style": "photographic"
            - "request": "pure color, clean, no patterns"
    - This rule takes priority when the user input clearly only focuses on color.
- For vague, mood-driven, or general inputs (e.g., "sleek look", "more business"):
    - Infer logical pattern, color, style, and request.
- For adding new design elements (e.g., "add a dragon"):
    - Extract "object_name" clearly, and describe the adjustment properly.
- If the user input is **vague, mood-driven, or general**:
    - Infer the most logical pattern, color, and request.
    - If still unclear, output 'use last' for pattern, color, style, and request.
- Always output specific, clear values.
- NEVER output 'unknown', 'none', 'null', 'default', or leave any field empty.
-- Always select **only ONE style** strictly from this list:
  enhance, anime, photographic, digital-art, comic-book, fantasy-art, line-art, analog-film, neon-punk, isometric, low-poly, origami, modeling-compound, cinematic, 3d-model, pixel-art, tile-texture.

--- STRICT INSTRUCTIONS ---
- Output **MUST be raw JSON only**, no text explanations.
- NEVER wrap the JSON in markdown code fences like ```json or ``` -- output pure JSON only.
- Do NOT add any text or comments before or after the JSON.
- Ensure the JSON is syntactically valid.

--- Output Format ---
Respond **STRICTLY in the following JSON format**, never output plain text or bullet points:
{{
  "adjustment": "...",
  "object_name": "...",
  "pattern": "...",
  "color": "...",
  "style": "...",
  "request": "..."
}}

--- Examples ---

Input: "Change color to red"
Output:
{{
  "adjustment": "Change the color to red.",
  "object_name": "use last",
  "pattern": "solid color",
  "color": "red",
  "style": "tile-texture",
  "request": "pure color, clean, no patterns"
}}

Input: "Replace the flames with blue lightning bolts"
Output:
{{
  "adjustment": "Replace the flames with blue lightning bolts.",
  "object_name": "flames",
  "pattern": "lightning bolts",
  "color": "blue",
  "style": "digital-art",
  "request": "energetic, intense"
}}

Input: "Add a dragon"
Output:
{{
  "adjustment": "Add a dragon to the design.",
  "object_name": "dragon",
  "pattern": "dragon motif",
  "color": "use last",
  "style": "anime",
  "request": "bold, mythical"
}}

Input: {input}
Output:
"""
) | llm



extract_edit_chain = PromptTemplate.from_template(
    """
You are an expert creative assistant. From the user's input, extract the detailed edit request **strictly in JSON format**.

--- Definitions ---
- 'Part' strictly refers to **physical car parts or specific car areas translated to car parts only**.
  - Valid outputs: "hood", "doors", "left_door", "right_door", "roof", "trunk", "rear", "front", "hood and doors"
  - **Do NOT return vague descriptions** like "side", "top", "back", "area", etc.
  - Use the following mappings for user-described locations:
    - "in the front of the car" → "hood"
    - "on the side of the car" → "doors"
    - "on the left side of the car" → "left_door"
    - "on the right side of the car" → "right_door"
    - "on top of the car" → "roof"
    - "on the back of the car" → "rear"
    - "in the front and side of the car" → "hood and doors"
  - If the user input does **not explicitly mention any of these car parts**, you MUST output 'use last' for 'Part'.
  - Never guess or invent a 'Part' if it's not clearly mentioned.

- 'Object_name' refers to the design element (e.g. lines, flames, dragon, stars, etc
- 'Pattern' should usually match or describe the object_name (e.g., lines → "lines"), otherwise, use: "use last"
- 'Color': If user explicitly states a color, extract it. Otherwise, use: "use last"
- 'Style': Always extract a **single value** from this fixed list:
  enhance, anime, photographic, digital-art, comic-book, fantasy-art, line-art, analog-film, neon-punk, isometric, low-poly, origami, modeling-compound, cinematic, 3d-model, pixel-art, tile-texture
  - Default to "photographic" unless user clearly indicates a different art style.
- 'Request' describes the **creative intent**. If the user expresses a mood (e.g., "feeling of speed"), translate that into a style request (e.g., "speed-inspired", "dynamic").

--- Extraction Rules ---
- **Special color-only change rule (high priority)**:
    - If the user input is **only about changing the color** (like 'change color to red', 'make it matte black') and does not mention pattern, object, mood, or parts:
        - Extract as:
            - "part": "use last"
            - "object_name": "use last"
            - "pattern": "solid color"
            - "color": "X"
            - "style": "tile-texture"
            - "request": "pure color, clean, no patterns"

- If the user input is **vague, mood-driven, or general**:
    - Infer the most logical pattern, color, and request.
    - If still unclear, output 'use last' for pattern, color, style, and request.

- Always output specific, clear values. NEVER output 'unknown', 'none', 'null', 'default', or leave any field empty.

--- STRICT INSTRUCTIONS ---
- Output **MUST be raw JSON only**, no text explanations.
- NEVER wrap the JSON in markdown code fences like ```json or ``` -- output pure JSON only.
- Do NOT add any text or comments before or after the JSON.
- Ensure the JSON is syntactically valid.

--- Output Format ---
Respond **STRICTLY in the following JSON format**, never bullet points or plain text:
{{
  "part": "...",
  "object_name": "...",
  "pattern": "...",
  "color": "...",
  "style": "...",
  "request": "..."
}}

--- Examples ---

Input: "Change color to red on the doors"
Output:
{{
  "part": "doors",
  "object_name": "use last",
  "pattern": "solid color",
  "color": "red",
  "style": "photographic",
  "request": "pure color, clean, no patterns"
}}

Input: "Add dragon on the left side in comic-book style"
Output:
{{
  "part": "left_door",
  "object_name": "dragon",
  "pattern": "dragon",
  "color": "use last",
  "style": "comic-book",
  "request": "bold, mythical"
}}

Input: "Remove the tropical leaves"
Output:
{{
  "part": "use last",
  "object_name": "tropical leaves",
  "pattern": "tropical leaves",
  "color": "use last",
  "style": "use last",
  "request": "remove elements, simplify"
}}

Input: "I want to add lines on the door, I want to have the feeling of speed"
Output:
{{
  "part": "doors",
  "object_name": "lines",
  "pattern": "lines",
  "color": "use last",
  "style": "photographic",
  "request": "speed-inspired, dynamic lines"
}}

Input: {input}
Output:
"""
) | llm





intent_chain = PromptTemplate.from_template("""
You are an AI assistant specializing in car wrap design.  
Your task is to detect the user's intent by carefully analyzing both the **user input** and the **session state**.

Session state contains:
- Last intent: {last_intent}
- Last pattern: {last_pattern}
- Last color: {last_color}
- Last style: {last_style}
- Last part: {last_part}
- Last object: {last_object}
- Last prompt: {last_prompt}
-Last URL: {last_image_url}

Current user input: {input}

Possible intents:
- initial: The user is starting a new design session, the session_state is empty, or this is the first round of the chat.
- adjust: The user wants to make minor or global modifications to the existing design (e.g., color, pattern, style, mood changes) across the entire image.
- edit: The user intends to make specific localized changes to particular parts (e.g., hood, door, roof, side), such as adding, removing, or changing elements on those areas.
- replace: The user wishes to discard the entire current design and start fresh with a completely different concept or theme.
- done: The user indicates that the design process is complete.

--- Clear and strict rules ---

1. **Session State Check First:**
   - If **session state is fully empty**, or **missing last_image_url**, classify as **initial**, no matter what the user says.
   - If session state has **only pattern, color, or style but no image URL, part, or object**, classify as **initial** (incomplete session).

2. **Explicit User Commands for Replace:**
   - Classify as **replace** only if the user says explicitly:
     - 'start over', 'scrap everything', 'replace the entire wrap', 'throw away this design', 'do a completely different design', 'new concept', 'let's restart', 'redo it'.

3. **Explicit 'done':**
   - Classify as **done** only if the user says explicitly: 'done', 'finished', 'perfect'.

4. **User mentions a specific part (hood, door, side, etc.):**
   - If session state has an active image (last_image_url exists), classify as **edit**.
   - Even if session is empty, classify as **initial**, unless the user explicitly says 'edit the current design'.

5. **User uses 'add', 'insert', 'include', 'also want', 'add more', 'replace', 'remove', or mood expressions like 'too bright', 'pop more', etc.:**
   - If session state has **last part or last object**, classify as **edit**.
   - If session state has only **last_image_url**, but no part/object, classify as **adjust**.
   - If session state is empty or incomplete (no image), classify as **initial**.
6.** When the user says 'change X to Y','replace X to Y',  and does NOT mention a specific car part, always classify as **adjust**.
    ** Only classify as **edit** if:
   - The user explicitly says 'on the hood', 'on the door', 'at the rear side',etc.
   - Or session state has 'last_part' and the user input is vague continuation.

6. **Vague user input (e.g., 'change color to red', 'make it more modern'):**
   - If session state has **last_image_url**, but user **DO NOT** make sprcific localized change to particular parts parts, classify as **adjust**.
   - If session statehas **last_image_url**, and the user make sprcific localized change to particular parts (e.g., hood, door, roof, side), classify as **edit**.
   - If session state is empty, classify as **initial**.

7. **Fallback default behavior:**
   - Prefer **adjust** if the user mentions global changes (color, pattern, mood) without parts.
   - Prefer **edit** only if the user explicitly mentions parts or specific localized elements.
   - Never classify as **replace** or **edit** unless conditions are clearly satisfied.

--- End of rules ---

--- STRICT INSTRUCTIONS ---
- Output **MUST be raw JSON only**, no text explanations.
- NEVER wrap the JSON in markdown code fences like ```json or ``` -- output pure JSON only.
- Do NOT add any text or comments before or after the JSON.
- Ensure the JSON is syntactically valid.

Respond only with the intent label:
initial, adjust, edit, replace, done

Examples:
1. User Input: "Let's begin designing a wrap for my new Tesla Model 3."
   Intent: initial

2. User Input: "Can we tweak the color scheme to include more red accents?"
   Intent: adjust

3. User Input: "Change the pattern from lines to circles throughout the entire wrap."
   Intent: adjust

4. User Input: "Remove all the tropical leaves."
   Intent: adjust

5. User Input: "Please change the design on the hood to feature a carbon fiber texture."
   Intent: edit

6. User Input: "Replace the stripes on the door with a geometric pattern."
   Intent: edit

7. User Input: "I want to scrap the current design and go with a completely different style."
   Intent: replace

8. User Input: "This design looks perfect. We're done here."
   Intent: done

9. User Input: "Let's add a cat"
   Session State: last intent was 'adjust' or 'initial' or 'replace'
   Intent: adjust

10. User Input: "Add a cat on the door"
    Session State: last pattern 'pixelated camouflage', last part is 'none'
    Intent: edit

11. User Input: "I also want to add stars"
    Session State: last intent was 'edit'
    Intent: edit

12. User Input: "Add stars on the hood"
    Intent: edit

13. User Input: "Make the color more pink"
    Session State: last intent was 'adjust'
    Intent: adjust

14. User Input: "Remove the stripes"
    Session State: last pattern 'geometric lines', no part,last intent was 'adjust'
    Intent: adjust

15. User Input: "Remove the stars on the hood"
    Intent: edit

16. User Input: "This design is too bright, make it more subtle"
    Intent: adjust

17. User Input: "Looks boring, can we add some energy?"
    Intent: adjust

18. User Input: "Make the door area more colorful"
    Intent: edit

19. User Input: "Can you make the pattern feel more dynamic?"
    Intent: adjust

20. User Input: "I want to start from scratch"
    Intent: replace

21. User Input: "Replace the dog with cat"
    Session State: last part 'hood', last intent 'edit',last object 'dog'
    Intent: edit

22. User Input: "Replace the logo on the door with our brand logo"
    Session State: last part 'door',last intent 'edit'
    Intent: edit

23. User Input: "Scrap everything and start over"
    Session State: any
    Intent: replace

24. User Input: "Let's do a completely new design"
    Intent: replace

25. User Input: "I do not like it. adding dog instead"
    Session State: last intent 'edit'
    Intent: edit

25. User Input: "I do not like it. adding dog instead"
    Session State: last intent 'adjust'
    Intent: adjust

Input: {input}
Session state: {session_state}
Intent:
""")






# ------------------- Prompt Generators -------------------
from langchain.prompts import PromptTemplate

from langchain.prompts import PromptTemplate

from langchain.prompts import PromptTemplate

text2image_prompt_chain = PromptTemplate.from_template(
    """
You are a prompt engineer for Stable Diffusion. Generate a high-quality prompt for 1024x1024 car wrap vinyl textures, suitable for large-format print.

Instructions:
- If **Pattern is 'solid color' or the user only requests color change**:
    - Focus only on **clean, seamless, pure {color} background**, no patterns, no decorations.
    - Use terms like "solid {color} background", "smooth vinyl wrap",  "1024x1024 vinyl wrap", "print-quality", "no reflections", "no lighting effects", "no patterns".

- Otherwise:
    - Emphasize {style} aesthetics.
    - Include {pattern} elements in {color}, designed to reflect: {request}.
    - Design must be symmetrical, seamless, photorealistic, and print-friendly.
    - Use terms like "vinyl wrap", "photorealistic", "studio lighting", "smooth gradients", "seamless {pattern} in {color}", "1024x1024 texture", "print-quality".
    - NEVER  include cars, texts, logos, or faces in the generated prompt.
    - NEVER use terms like "car wrap".

Few-shot examples:

Input: Pattern: solid color, Color: green, Style: photographic, Request: pure color, clean, no patterns  
Prompt: "Solid green color, flat and uniform vinyl wrap texture, no lighting, no shading, no gradient, no reflections, no texture, no patterns, no logos, pure green background, print-quality, seamless,1024x1024"

Input: Pattern: flames, Color: red and orange, Style: digital-art, Request: aggressive, emphasize heat and speed  
Prompt: "Aggressive red and orange flame pattern, digital-art style, symmetrical and seamless design, photorealistic vinyl texture, print-quality,1024x1024"

Input: Pattern: geometric lines, Color: silver and black, Style: digital-art, Request: futuristic, high-tech luxury feel  
Prompt: "Futuristic high-tech style with silver and black geometric lines, digital-art aesthetics, symmetrical and seamless pattern, print-quality,1024x1024"

Now generate the prompt for:
- Pattern: {pattern}
- Color: {color}
- Style: {style}
- Request: {request}

Prompt:
"""
) | llm





img2img_adjust_prompt_chain = PromptTemplate.from_template(
    """
You are a prompt engineer for Stable Diffusion's image-to-image model.

Modify the existing car wrap design based on the following extracted information:

- Adjustment: {adjustment}
- Pattern: {pattern}
- Color: {color}
- Style: {style}
- Object_name: {object_name}
- Request: {request}

Instructions:
- If **Pattern is 'solid color'**:
    - Describe only changing the entire wrap to a **solid {color} background**, no patterns, no decorations.
    - Use terms like "replace entire design with solid {color} background", "smooth vinyl wrap", "print-quality","no reflections", "no lighting effects", "no patterns.
- If **Adjustment includes 'add' or 'include'**:
    - Clearly describe the new element to be added (e.g., "add a dragon").
    - Specify its placement, size, and integration with the existing design.
    - Use descriptors like "vinyl wrap," "photorealistic, and "smooth gradients" to enhance print quality.
- Describe clearly the visual change in {adjustment}.
- If {object_name} or {pattern}  needs remove, delete, or erase, describe only the removal of the specified object_name or pattern, ensuring the design becomes cleaner and simpler.
- If adding/modifying, describe {object_name} and/or {pattern} in {color}, using {style} style, with a {request} mood.
- If {object_name} is 'none' or empty, focus only on {pattern}.
- If {pattern} is 'none' or empty, focus only on {object_name}.
- Always ensure the design is seamless, symmetrical, photorealistic, and suitable for large-format vinyl wrap printing.
- Ensure the design is symmetrical, seamless, photorealistic, and vinyl wrap texture.
- NEVER invent elements not present in the extracted info.
- NEVER  include cars, texts, logos, or faces in the generated prompt, unless they are explicitly specified.
- NEVER use terms like "car wrap".


Few-shot examples:
Input:
- Adjustment: Change the red flames to blue lightning bolts.
- Object_name: flames
- Pattern: lightning bolts
- Color: blue
- Style: digital-art
- Request: energetic, intense
Prompt:
"Blue lightning bolts replacing red flames, symmetrical, photorealistic texture, print-quality,1024x1024"

Input:
- Adjustment: Transform the camouflage pattern into a sleek matte black finish.
- Object_name: camouflage
- Pattern: matte black finish
- Color: matte black
- Style: tile-texture
- Request: elegant, luxurious
Prompt:
"Sleek matte black finish replacing camouflage pattern, smooth gradients, tile-texture texture, 1024x1024 vinyl wrap, print-quality,no shading, no gradient, no reflections, no lighting effects, no texture,no patterns"

Input:
- Adjustment: Add whimsical elements and soften the texture.
- Object_name: hearts and clouds
- Pattern: hearts and clouds
- Color: pastel pink and blue
- Style: anime
- Request: cute, dreamy
Prompt:
"Whimsical hearts and clouds in pastel pink and blue, enhancing cuteness and dreamy mood, anime style, symmetrical and seamless, photorealistic texture, 1024x1024 vinyl wrap, studio lighting, smooth gradients, print-quality"

Input:
- Adjustment: Add a dragon to the design.
- Object_name: dragon
- Pattern: dragon
- Color: red and gold
- Style: traditional Asian
- Request: bold, mythical
Prompt:
"Red and gold traditional Asian dragon, bold and mythical mood, symmetrical and seamless layout, photorealistic texture, 1024x1024 vinyl wrap, studio lighting, smooth gradients, print-quality"

Now generate the prompt:
Prompt:
"""

) | llm





inpaint_prompt_chain = PromptTemplate.from_template(
    """
You are a prompt engineer for Stable Diffusion's inpainting model.
Generate a concise, high-quality, print-ready inpainting prompt to modify an existing car wrap design.


Use strictly the extracted info below:
- Object_name: {object_name}
- Pattern: {pattern}
- Color: {color}
- Style: {style}
- Request: {request}


Instructions:
- If **Pattern is 'solid color'**:
    - Describe only changing the specific area to a **solid {color} background**, no patterns, no decorations,no reflections, no lighting effects, no patterns.
    - Use terms like "replace selected area with solid {color} background", "smooth vinyl wrap texture", "1024x1024 vinyl wrap,no patterns, no decorations,no reflections, no lighting effects, no patterns.".
- If {object_name} or {pattern} needs to be removed, deleted, or erased:
    - Describe only the removal. Do not modify the rest of the design.
    - Result should be cleaner and simpler.

- If **adding** new content (e.g., "add a dog", "add stars"):
    - Clearly describe the added object/pattern **within the masked area only**.
    - Never modify areas outside the masked zone.
    - Use phrasing like "add {object_name} in {color}, {style} style, within masked area", "only modify masked region".

- If modifying, describe {object_name} and/or {pattern} in {color}, using {style} style, with a {request} mood.
- If {object_name} is 'none' or empty, focus only on {pattern}.
- If {pattern} is 'none' or empty, focus only on {object_name}.
- Always ensure the design is seamless, symmetrical, photorealistic, and suitable for large-format vinyl wrap printing.
- NEVER mention car parts like 'hood', 'door', 'roof', or 'side'.
- The inpainted result must **fit naturally and precisely within the masked region**. Do not generate content that exceeds or ignores the masked area boundaries.
- You may generate **any content requested by the user** based on the extracted inputs.
- However, the output must **never include cars or any car parts**, such as hoods, wheels, bumpers, windows, etc.
- **Text, logos, faces, people, or other specific elements are allowed if explicitly requested** in the input.
- Avoid to include any words about car, varhicle in the generated prompt, unless they are explicitly specified.
- NEVER use terms like "car wrap".


Few-shot examples:
Input:
- Object_name: none
- Pattern: tropical leaves
- Color: red orange
- Style: digital-art
- Request: remove elements, simplify  
Prompt:
"Remove red orange tropical leaves, simplify the design, maintain seamless photorealistic texture, print-quality, digital-art style,1024x1024, no car, no vehicle, no reflections, no lighting effects,fits masked area"

Input:
- Object_name: dragon emblem
- Pattern: flames
- Color: red
- Style: comic-book
- Request: bold, mythical  
Prompt:
"Red dragon emblem with subtle flames, blending seamlessly into the masked area, photorealistic texture, print-quality, 1024x1024,comic-book style,no reflections, no lighting effects"

Input:
- Object_name: none
- Pattern: geometric pattern
- Color: silver
- Style: futuristic
- Request: sleek, modern  
Prompt:
"Silver geometric pattern, symmetrical design, seamless and photorealistic texture for vinyl wrap, fits masked region, 1024x1024, print-quality, futuristic style,no reflections, no lighting effects"

Input:
- Object_name: tiger face
- Pattern: stripes
- Color: orange
- Style: fantasy-art
- Request: aggressive, wild  
Prompt:
"Orange tiger face with matching stripes, blending seamlessly within the masked area, photorealistic texture, 1024x1024 vinyl wrap, print-quality, fantasy-art style,no reflections, no lighting effects"

Now generate the prompt:
Prompt:
"""
) | llm

#------------guidance chain ---------------

guidance_chain = PromptTemplate.from_template("""
You are a car wrap design assistant.

Your task is to suggest what the user can do next, based on:
- The last detected intent: {last_intent}
- The current session state (JSON): {session_state}
- The recent chat history (compressed summary): {history}

--- Intent Definitions ---
- **Adjust**: Make global modifications to the entire design — such as changing overall color, pattern, or style. These apply across the whole vehicle.
- **Edit**: Make localized visual enhancements on high-impact areas — specifically the **doors**, **hood**, or **both**. These are the best spots for focused detailing and creative accents. All edit suggestions should involve only these parts.
- **Replace**: Start over with a completely new design concept, discarding the current one.
- **Done**: The user is satisfied with the design and ready to finalize.

--- Output Rules ---
- Provide **exactly one example per intent**.
- Use `session_state` fields like `pattern`, `color`, `style`, `object_name`, and `part` to personalize suggestions.
- Use `history` to reflect past design steps or requests.
- For **edit**, always focus on **hood**, **doors**, or **both**. Present this as an opportunity for maximum visual impact — not a limitation.
- Keep the tone user-friendly, confident, and helpful.

--- Few-Shot Example ---

Input:
- last_intent: "edit"
- session_state: {{
    "pattern": "floral",
    "color": "blue",
    "style": "photorealistic",
    "object_name": "flower",
    "part": "door"
  }}
- history: "User added floral elements on the door."

Output:
{{
  "adjust_examples": ["Shift the entire design to a soft pastel floral tone"],
  "edit_examples": ["Refine the flower pattern on the hood for a bolder effect"],
  "replace_examples": ["Begin a new concept with angular geometric shapes in black"],
  "done_examples": ["Looks complete—let’s move forward to production"]
}}

Now generate suggestions using:
- last_intent: {last_intent}
- session_state: {session_state}
- history: {history}

Respond strictly in this JSON format:
{{
  "adjust_examples": ["..."],
  "edit_examples": ["..."],
  "replace_examples": ["..."],
  "done_examples": ["..."]
}}
""") | llm



#-------------helper------------

def extract_kv(text, keys):
    import re
    result = {k.lower(): "" for k in keys}
    key_pattern = "|".join(keys)
    pattern = re.compile(rf"^\s*[-•]?\s*({key_pattern})\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
    for match in pattern.finditer(text):
        key, value = match.group(1).strip().lower(), match.group(2).strip()
        result[key] = value
    return result


def merge_with_session_state(extracted_info: dict, session_state: dict):
    merged = extracted_info.copy()
    merged["part"] = None  # Always enforce null
    # Use last pattern
    if merged.get("pattern", "").lower() == "use last":
        merged["pattern"] = session_state.get("last_pattern")
    if merged.get("object_name", "").lower() == "use last":
        merged["object_name"] = session_state.get("last_object_name")
    # Use last color
    if merged.get("color", "").lower() == "use last":
        merged["color"] = session_state.get("last_color")
    # Use last style
    if merged.get("style", "").lower() == "use last":
        merged["style"] = session_state.get("last_style")  # You can set your default style here
    # Use last request (optional)
    if merged.get("request", "").lower() == "use last":
        merged["request"] = session_state.get("last_request")
    return merged



def merge_edit_with_session_state(extracted_info, session_state):
    merged = extracted_info.copy()
    if merged.get("part", "").lower() == "use last":
        merged["part"] = session_state.get("last_part")
    if merged.get("color", "").lower() == "use last":
        merged["color"] = session_state.get("last_color")
    if merged.get("pattern", "").lower() == "use last":
        merged["pattern"] = session_state.get("last_pattern")
    if merged.get("style", "").lower() == "use last" or not merged.get("style"):
        merged["style"] = session_state.get("last_style")  # Default style
    if merged.get("request", "").lower() == "use last":
        merged["request"] = session_state.get("last_request")
    if merged.get("object_name", "").lower() == "use last":
        merged["object_name"] = session_state.get("last_object_name")
    return merged



# ------------------ Reasoning Tools ------------------

# ---------------------------- Reasoning Functions ----------------------------

def detect_intent(user_input, forced_intent=None):
    chat_history = short_term_memory.load_memory_variables({})["chat_history"]

    # Ensure we unpack session_state correctly
    return (intent_chain | llm).invoke({
        "chat_history": chat_history,
        "input": user_input,
        "last_color": session_state.get("last_color", "none"),
        "last_intent": session_state.get("last_intent", "none"),
        "last_object": session_state.get("last_object", "none"),
        "last_part": session_state.get("last_part", "none"),
        "last_pattern": session_state.get("last_pattern", "none"),
        "last_prompt": session_state.get("last_prompt", "none"),
        "last_style": session_state.get("last_style", "none"),
        "last_image_url": session_state.get("last_image_url", "none"),
        "session_state": session_state  # You can still include the whole for redundancy if your prompt uses it generically
    }).content.strip()



def extract_design(user_input: str):
    raw_text = extract_design_chain.invoke({"input": user_input}).content.strip()
    return extract_kv(raw_text, ["pattern", "color", "style", "request"])


def extract_adjust(user_input: str):
    # Call the prompt chain (which returns an AIMessage with .content)
    raw_text = extract_adjustment_chain.invoke({"input": user_input}).content.strip()
    print(f"[ExtractEdit Raw JSON]: {raw_text}")
    
    # Debugging log to verify raw output
    print(f"[ExtractAdjust Raw JSON]: {raw_text}")
    
    # Parse the content directly as JSON
    try:
        extracted = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON returned by extract_adjustment_chain:\n{raw_text}\nError: {str(e)}")

    # Enforce 'part' is always None for adjust intent
    extracted["part"] = None

    # Optional: enforce that required fields are present and non-empty
    required_keys = ["adjustment", "object_name", "pattern", "color", "style", "request"]
    for key in required_keys:
        if key not in extracted or not extracted[key].strip():
            raise ValueError(f"Missing or empty '{key}' in extracted adjustment: {extracted}")

    # Merge with session state if needed
    return merge_with_session_state(extracted, session_state)



def extract_edit(user_input: str):
    # Step 1: Invoke the LLM chain
    raw_text = extract_edit_chain.invoke({"input": user_input}).content.strip()
    print(f"[ExtractEdit Raw JSON]: {raw_text}")  # Debug

    # Step 2: Parse raw output as JSON
    try:
        extracted = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON returned by extract_edit_chain:\n{raw_text}\nError: {str(e)}")

    # Step 3: Enforce required fields are present and not empty (except "use last")
    required_keys = ["part", "object_name", "pattern", "color", "style", "request"]
    for key in required_keys:
        if key not in extracted:
            raise ValueError(f"Missing key '{key}' in extracted edit: {extracted}")
        val = extracted[key].strip() if isinstance(extracted[key], str) else str(extracted[key])
        if not val:
            raise ValueError(f"Empty value for key '{key}' in extracted edit: {extracted}")
    return merge_edit_with_session_state(extracted, session_state)


def generate_text2image_prompt(pattern: str, color: str, style: str, request: str):
    if not pattern or pattern.strip().lower() in ["", "none", "unknown"]:
        raise ValueError("Invalid pattern.")
    if not color or color.strip().lower() in ["", "none", "unknown"]:
        raise ValueError("Invalid color.")
    if not style or style.strip().lower() in ["", "none", "unknown"]:
        raise ValueError("Invalid style.")
    if not request or request.strip().lower() in ["", "none", "unknown"]:
        raise ValueError("Invalid request.")
    
    return text2image_prompt_chain.invoke({
        "pattern": pattern,
        "color": color,
        "style": style,
        "request": request
    }).content.strip()



def generate_img2img_prompt(adjustment: str, pattern: str, color: str, style: str, request: str, object_name=None):
    if not adjustment or adjustment.strip().lower() in ["", "none", "unknown"]:
        raise ValueError("Invalid adjustment.")
    if not pattern or pattern.strip().lower() in ["", "none", "unknown"]:
        raise ValueError("Invalid pattern.")
    if not color or color.strip().lower() in ["", "none", "unknown"]:
        raise ValueError("Invalid color.")
    if not style or style.strip().lower() in ["", "none", "unknown"]:
        raise ValueError("Invalid style.")
    if not request or request.strip().lower() in ["", "none", "unknown"]:
        raise ValueError("Invalid request.")
    
    return img2img_adjust_prompt_chain.invoke({
        "adjustment": adjustment,
        "object_name": object_name,
        "pattern": pattern,
        "color": color,
        "style": style,
        "request": request
    }).content.strip()


def generate_inpainting_prompt( pattern: str, color: str, style: str, request: str, object_name=None):
    if not object_name or object_name.strip().lower() in ["", "none", "unknown"]:
        raise ValueError("Invalid object.")
    if not pattern or pattern.strip().lower() in ["", "none", "unknown"]:
        raise ValueError("Invalid pattern.")
    if not color or color.strip().lower() in ["", "none", "unknown"]:
        raise ValueError("Invalid color.")
    if not style or style.strip().lower() in ["", "none", "unknown"]:
        raise ValueError("Invalid style.")
    if not request or request.strip().lower() in ["", "none", "unknown"]:
        raise ValueError("Invalid request.")
    
    return inpaint_prompt_chain.invoke({
        "object_name": object_name,
        "pattern": pattern,
        "color": color,
        "style": style,
        "request": request
    }).content.strip()


# ---------------------------- Tools Setup ----------------------------

intent_tool = Tool.from_function(
    detect_intent,
    name="DetectIntent",
    description="Detect user's intent using memory."
)

extract_design_tool = Tool.from_function(
    extract_design,
    name="ExtractDesignInfo",
    description="Extract design info into structured dict from user input."
)

extract_adjust_tool = Tool.from_function(
    extract_adjust,
    name="ExtractAdjustInfo",
    description="Extract adjust info and fill missing pattern, color, style, and request from session state."
)

extract_inpainting_tool = Tool.from_function(
    extract_edit,
    name="ExtractInpaintingInfo",
    description="Extract inpainting info and fill missing part, color, pattern, style, and request from session state."
)

generate_text2image_prompt_tool = StructuredTool.from_function(
    generate_text2image_prompt,
    name="GenerateText2ImagePrompt",
    description="Generate Stability AI text-to-image prompt from extracted design info."
)

generate_img2img_prompt_tool = StructuredTool.from_function(
    generate_img2img_prompt,
    name="GenerateImg2ImgPrompt",
    description="Generate Stability AI img2img prompt from extracted adjust info."
)

generate_inpainting_prompt_tool = StructuredTool.from_function(
    generate_inpainting_prompt,
    name="GenerateInpaintingPrompt",
    description="Generate Stability AI inpainting prompt from extracted edit info."
)



# ------------------- Planning & Reflection -------------------

planning_chain = PromptTemplate.from_template(
    """
You are a car wrap design planning assistant.

Your task is to carefully analyze the **user input** and **session state**, and plan the next steps.

--- Decision Logic ---
1. Always detect the user's **intent**, strictly from these options:
   - initial
   - adjust
   - edit
   - replace
   - done

2. Follow these strict rules:
- If session state is empty or 'last_image_url' is missing → intent is 'initial'.
- If the user explicitly says 'start over', 'new design', 'replace everything' → intent is 'replace'.
- If the user explicitly says 'done', 'finished', 'perfect' → intent is 'done'.
- If the user mentions specific parts (hood, doors, side) → intent is 'edit'.
- If the user asks for global change (color, pattern, style, mood) → intent is 'adjust'.

3. After deciding the intent, carefully list the tool steps.

--- Output Rules ---
- You must respond strictly in **pure JSON format**.:
{{
  "intent": "<one of: initial, adjust, edit, replace, done>",
  "tool_steps": [
    "<tool name>",
    "<tool name>",
    ...
  ],
  "summary": "<summarize what will be done>"
}}
- Do NOT wrap the output in any ```json block or text.
- Do NOT add any explanation before or after the JSON.
--- Example ---

Example 1 (Initial):
User input: "Let's create a new wrap with flames."
Session state: (empty)

Response:
{{
  "intent": "initial",
  "tool_steps": [
    "DetectIntent",
    "ExtractDesignInfo",
    "GenerateText2ImagePrompt"
  ],
  "summary": "Start a new design session by extracting design info and generating a text-to-image prompt."
}}

Example 2 (Adjust):
User input: "Make the color more vibrant and pop."
Session state: last_image_url exists

Response:
{{
  "intent": "adjust",
  "tool_steps": [
    "DetectIntent",
    "ExtractAdjustInfo",
    "GenerateImg2ImgPrompt"
  ],
  "summary": "User wants to make global adjustments to the existing design, such as changing the color or pattern."
}}

Example 3 (Edit):
User input: "Add a dragon on the hood."
Session state: last_image_url exists

Response:
{{
  "intent": "edit",
  "tool_steps": [
    "DetectIntent",
    "ExtractInpaintingInfo",
    "GenerateInpaintingPrompt"
  ],
  "summary": "User wants to modify specific car parts (like hood) by adding or changing elements."
}}

Example 4 (Replace):
User input: "Let's start over with a new design concept."
Session state: last_image_url exists

Response:
{{
  "intent": "replace",
  "tool_steps": [
    "DetectIntent",
    "ExtractDesignInfo",
    "GenerateText2ImagePrompt"
  ],
  "summary": "User wants to discard the current design and start a new concept from scratch."
}}


--- Now plan carefully based on below ---

Chat History:
{chat_history}

User Input:
{input}

Session state:
{session_state}

Respond:
"""
) | llm


def extract_intent_from_plan(plan_text):
    try:
        plan_json = json.loads(plan_text)  # Always required
        return plan_json.get("intent", "unknown").lower()
    except json.JSONDecodeError as e:
        print(f"[Error]: Failed to parse plan as JSON:\n{plan_text}\nError: {e}")
        return "unknown"



#sample refelction
reflection_chain = PromptTemplate.from_template(
    """
You are a disciplined reflection agent for car wrap design.

Your task is simple:
- Compare the agent's **detected intent** and **executed tool steps** against the **planned intent and steps**.

--- Decision Rules ---

🔁 Track retry attempts using the conversation context. If this is the **third attempt** (after two retries), you MUST choose 'clarify' instead of continuing to retry, unless the situation is perfectly resolvable.


✅ Accept if:
- The detected intent matches the planned intent AND
- The executed steps exactly follow the planned tool sequence.

🧠 Override the plan and accept the executed version if:
- The detected intent or executed steps do not match the plan.
- AND you are confident the executed version better reflects the user input.
- Use the user input, chat history, last detected intent, and prior session context (e.g., last image, last part, last object) to make this decision.
- In this case, respond with "accept" and provide a short justification inside a comment.

❌ Retry if:
- There is a mismatch in plan vs. execution AND
- You can identify a mistake but CANNOT confidently determine a better alternative from the input.

❓ Clarify if:
- The user input is too vague OR
- You cannot confidently choose between multiple possible intents or flows based on the input and chat history.
- You have retried twice and are now on the third attempt. In this case, your clarification **must address the core ambiguity that led to previous retries**, without mentioning prior attempts — for example:
  - "Are you trying to change a specific part or the entire design?"
  - "Should this replace the existing design or just modify its color or elements?"
--- Review Context ---

User Input:
{input}

Chat History:
{chat_history}

Planned Intent:
{plan_intent}

Planned Steps:
{plan_steps}

Executed Steps:
{executed_steps}

Merged Extraction:
- Pattern: {pattern}
- Color: {color}
- Style: {style}
- Part: {part}
- Object_name: {object_name}
- Request: {request}

Prompt:
{prompt}

Respond strictly in JSON format only:
Respond strictly in JSON format only:
{{
  "result": "accept"
}}

OR:
{{
  "result": "retry",
  "reason": "Explain clearly why retry is needed."
}}

OR:
{{
  "result": "clarify",
  "reason": "Explain why clarification is needed.",
  "hint": "Could you clarify your request? For example, 'sleek geometric lines in silver' or 'floral pattern in pastel pink'."
}}
"""
) | llm








# ------------------ Agent Setup ------------------

reasoning_tools = [
    intent_tool,   # comment this, if Only allow extraction and generation tools, block intent tool
    extract_design_tool,
    extract_adjust_tool,
    extract_inpainting_tool,
    generate_text2image_prompt_tool,
    generate_img2img_prompt_tool,
    generate_inpainting_prompt_tool
]

# Create a prompt with a system message
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

system_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a highly disciplined car wrap design reasoning agent.

You must:
- Independently reason and detect the user's intent using the available tools.
- Carefully select the correct extraction and generation tools **based on your own detected intent**, not based on any external plan.
- You are responsible for reasoning, tool calling, and generating the car wrap prompt.
- Reflection agent will later compare your steps with the planner's plan.
- You must not read or reference the planner's plan directly.
- You must document your steps via tool calls only.

--- Rules for Execution ---
1. Always first use `DetectIntent` tool to detect the intent.
2. Based on the detected intent:
   - If **Intent = 'initial'**:
     - Use `ExtractDesignInfo` to extract design information.
     - Then use `GenerateText2ImagePrompt` to generate the design prompt.
   - If **Intent = 'adjust'**:
     - Use `ExtractAdjustInfo` to extract adjustment information.
     - Then use `GenerateImg2ImgPrompt` to generate the adjustment prompt.
   - If **Intent = 'edit'**:
     - Use `ExtractInpaintingInfo` to extract edit information.
     - Then use `GenerateInpaintingPrompt` to generate the localized edit prompt.
   - If **Intent = 'replace'**:
     - Use `ExtractDesignInfo` to extract new design information.
     - Then use `GenerateText2ImagePrompt` to generate the new design prompt.
   - If **Intent = 'done'**:
     - Conclude the session without any tool call.

3. Never skip any tool step.
4. Always use the output of the extraction tool as the direct input to the generation tool.
5. Never invent or guess any fields.
6. Never write or modify the prompt yourself. Always use the correct generation tool.

--- Strict Output Format ---
Always respond strictly in the following JSON format:
{{
  "intent": "<your detected intent>",
  "prompt": "<your generated car wrap prompt>",
  "color": "<optional, main color if applicable>",
  "style": "<optional, style if applicable>",
  "pattern": "<optional, pattern if applicable>",
  "object_name": "<optional, object name if applicable>",
  "part": "<optional, car part if applicable>",
  "request": "<optional, request if applicable>"
}}

- If any field is not applicable, set it to 'null' or omit.
- Never skip 'intent'.
- Never reply with only 'prompt'.
- Never include comments or explanations outside of the JSON.
"""),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad")
])



#agent = create_openai_functions_agent(llm=llm, tools = reasoning_tools, prompt = system_prompt)
#agent_executor = AgentExecutor(agent=agent, tools=reasoning_tools, verbose=True)


import os
import re
import json
import random
from langchain.memory import ConversationBufferMemory

# Initialize memory and session state
short_term_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
session_state = {
    "last_image_url": None,
    "last_prompt": None,
    "last_pattern": None,
    "last_color": None,
    "last_request": None,
    "last_part": None
}

# Create agent and executor
agent = create_openai_functions_agent(llm=llm, tools=reasoning_tools, prompt=system_prompt)
agent_executor = AgentExecutor(agent=agent, tools=reasoning_tools, verbose=True, return_intermediate_steps=True)

# Clean output

def clean_agent_output(output_text):
    cleaned_text = re.sub(r"```json\s*([\s\S]*?)\s*```", r"\1", output_text.strip())
    cleaned_text = re.sub(r"```([\s\S]*?)\s*```", r"\1", cleaned_text.strip())
    return cleaned_text.strip()

# Path helper
def safe_output_path(folder_name, file_name, base_dir=None):
    if base_dir is None:
        base_dir = os.getcwd()
    output_dir = os.path.join(base_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, file_name)

def run_agent_par_with_auto_retry(max_retries=5):
    seed = random.randint(0, 2**32 - 1)
    print(f"Random seed: {seed}")
    print("\n--- AI Car Wrap Agent with Reflection Loop (Auto-Retry) ---\n")

    print(example_chain.invoke({}))

    rounds = 1
    while True:
        user_input = input("\nYou: ")
        if user_input.strip().lower() == "done":
            print("Session complete.")
            break

        plan = planning_chain.invoke({
            "chat_history": short_term_memory.load_memory_variables({})["chat_history"],
            "input": user_input,
            "session_state": session_state
        }).content
        print(f"[Planning]: {plan}")
        plan_json = json.loads(plan)

        detected_intent = extract_intent_from_plan(plan)
        print(f"[Detected Intent (From Plan)]: {detected_intent}")
        

        retries = 0
        reflection_reason = None

        while retries < max_retries:
            scratchpad = [
                {"role": "system", "content": f"""You are a car wrap design reasoning agent.

Current Session State:
{json.dumps(session_state, indent=2)}

Rules:
- Always detect intent first.
- Then extract info.
- Then generate prompt.
- Use tools only. Never invent or skip.
- Always fix the last reflection feedback if any.

Proceed carefully.
"""}
            ]

            if reflection_reason:
                scratchpad.append({
                    "role": "system",
                    "content": f"Reflection feedback from last attempt: {reflection_reason}. Fix this specific issue carefully in your next attempt. Do NOT repeat the mistake."
                })

            result = agent_executor.invoke({
                "input": user_input,
                "detected_intent": detected_intent,
                "agent_scratchpad": scratchpad
            })

            print(f"[Reasoning Result]: {result['output']}")

            executed_steps_formatted = []
            for action, observation in result['intermediate_steps']:
                executed_steps_formatted.append({
                    "tool": action.tool,
                    "input": action.tool_input,
                    "output": observation
                })

            print("[Executed Steps]:")
            for step in executed_steps_formatted:
                print(step)

            cleaned_output = clean_agent_output(result['output'])

            try:
                extracted_info = json.loads(cleaned_output)
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed: {e}. Retrying...")
                retries += 1
                continue

            part_for_reflection = ""
            if extracted_info.get("intent") == "edit":
                part_for_reflection = extracted_info.get("part", "")

            reflection_json = reflection_chain.invoke({
                "plan_intent": plan_json["intent"],
                "plan_steps": plan_json["tool_steps"],
                "input": user_input,
                "intent": extracted_info.get("intent", "unknown"),
                "prompt": extracted_info.get("prompt", ""),
                "pattern": extracted_info.get("pattern", ""),
                "color": extracted_info.get("color", ""),
                "style": extracted_info.get("style", ""),
                "object_name": extracted_info.get("object_name", ""),
                "part": extracted_info.get("part", ""),
                "request": extracted_info.get("request", ""),
                "last_image_url": session_state.get("last_image_url", None),
                "executed_steps": executed_steps_formatted,
                "chat_history": short_term_memory.load_memory_variables({})["chat_history"]
            }).content.strip()

            print(f"[Reflection Decision]: {reflection_json}")

            try:
                reflection_result = json.loads(clean_agent_output(reflection_json))
            except json.JSONDecodeError:
                print(f"Reflection unrecognized. Forcing retry.\n{reflection_json}")
                retries += 1
                continue

            reflection_status = reflection_result.get("result", "retry")

            if reflection_status == "retry":
                retries += 1
                reflection_reason = reflection_result.get("reason", "Unknown mistake detected.")
                print(f"Reflection suggests retrying... Attempt {retries}/{max_retries} - Reason: {reflection_reason}")
                continue

            if reflection_status == "clarify":
                hint = reflection_result.get("hint", "Could you clarify your request?")
                print(f"❓ Reflection suggests clarification. Hint: {hint}")
                break

            if reflection_status == "accept":
                print("Reflection accepted. Updating session state.")
                intent = extracted_info.get("intent")
                style = extracted_info.get("style")
                prompt = extracted_info.get("prompt")

                if intent in ["initial", "replace"]:
                    output_path = safe_output_path("image", f"{seed}_{rounds}_initial.png")  
                    txt2img.generate_background_image(prompt=prompt, api_key=stability_api_key, output_path=output_path, style_type=style, seed=seed)
                    print("Here is what you can do next:")
                    gudiance = guidance_chain.invoke({"last_intent":intent,"session_state":session_state,"history": short_term_memory.load_memory_variables({})["chat_history"]})
                    print(gudiance.content)
                    session_state['last_image_url'] = output_path
                elif intent == 'adjust':
                    input_image_path = session_state['last_image_url']
                    output_path = safe_output_path("image", f"{seed}_{rounds}_adjust.png")
                    img2img.generate_img2img_adjust(input_image_path=input_image_path, prompt=prompt, output_path=output_path, api_key=stability_api_key, style_preset=style, seed=seed)
                    print("Here is what you can do next:")
                    gudiance = guidance_chain.invoke({"last_intent":intent,"session_state":session_state,"history": short_term_memory.load_memory_variables({})["chat_history"]})
                    print(gudiance.content)
                    session_state['last_image_url'] = output_path
                elif intent == 'edit':
                    input_image_path = session_state['last_image_url']
                    output_path = safe_output_path("image", f"{seed}_{rounds}_edit.png")
                    edit_part = extracted_info.get("part")
                    if edit_part == "hood":
                        mask_image_path = safe_output_path("mask", "hood.png")
                    elif edit_part in ["doors", "left_door", "right_door"]:
                        mask_image_path = safe_output_path("mask", "doors.png")
                    else:
                        mask_image_path = safe_output_path("mask", "doors_hood.png")
                    inp.generate_background_image_inpainting(prompt=prompt, api_key=stability_api_key, save_path=output_path, init_image_path=input_image_path, mask_image_path=mask_image_path, style_preset=style, seed=seed)
                    print("Here is what you can do next:")
                    gudiance = guidance_chain.invoke({"last_intent":intent,"session_state":session_state,"history":short_term_memory.load_memory_variables({})["chat_history"]})
                    print(gudiance.content)
                    session_state['last_image_url'] = output_path

                session_state["last_prompt"] = extracted_info.get("prompt", "")
                for key in ["color", "pattern", "style", "object_name", "request", "intent"]:
                    if extracted_info.get(key) and extracted_info[key].lower() not in ["unknown", "null"]:
                        session_state[f"last_{key}"] = extracted_info[key]

                if intent == "edit" and extracted_info.get("part") and extracted_info["part"].lower() not in ["unknown", "null"]:
                    session_state["last_part"] = extracted_info["part"]
                else:
                    session_state["last_part"] = None

                print(f"[Updated Session State]: {session_state}")
                break

        if retries >= max_retries:
            print(f"❗ Exceeded max retries ({max_retries}). Please revise your input.")

        rounds += 1

if __name__ == "__main__":
    run_agent_par_with_auto_retry()
