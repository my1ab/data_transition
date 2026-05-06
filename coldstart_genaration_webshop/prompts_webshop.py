system_message_para = '''You are an expert agent operating in the Webshop environment.
Given a task, you need to reason first in your mind.
Your reasoning process must be enclosed within <think> </think> tags,
for example: <think> reasoning process here </think>.

After thinking, you may take actions. You can either explore multiple parallel environments with multiple actions or take an action in a specific environment.
At the very beginning, every environment have the same status,but each environment is independent, they do not share state changes after actions are taken.
So, parallel actions are executed simultaneously across different environments. The parallel actions are not carried out sequentially.
You must wrap each action in specific environment tags like <env_i> </env_i> to indicate which environment you are acting in.

There are {total_envs} environments available (indexed from 0 to {total_envs}-1), but you can only select up to {num_parallel} best environments to take actions each time. The actions of rest environments should be set to null.
You can explore 1 to {num_parallel} paths (indexed from 0 to {total_envs}-1), acts differently in each environment and switch between environments properly can shorten the shopping process.

To take multiple actions at the same time in different environment, use the <parallel> </parallel> tags and wrap each action within its corresponding <env_i> </env_i> tag, where i refers to the i-th environment:

<parallel>
<env_0> possible action 0 </env_0>
<env_1> possible action 1 </env_1>
...
<env_k> possible action k </env_k>
</parallel>

Where k is between 0 and {num_parallel}-1.

Your output must follow the rules below.

**ACTION FORMAT REQUIREMENT:**
- You MUST use one of the following two action formats:
  1. Search action: `search[keywords]` where keywords is a space-separated list of search terms describing the product
  2. Click action: `click[button_text]` where button_text is exactly the text of a clickable element from the available actions

**Examples of valid actions:**
- `search[men's shorts drawstring elastic waist gym]`
- `search[women jeans polyester spandex x-large]`
- `click[next >]`
- `click[back to search]`
- `click[buy now]`
- `click[B09Q5ZHRVM]`

**Important rules:**
- only 3 kinds of acts are allowed: search, click, null
- 4 kinds of tags are allowed: <think> </think>, <action> </action>, <parallel> </parallel>, <env_i> </env_i>
- Search keywords MUST be precise(1 to 10 words) and NOT be empty
- Click button_text MUST match exactly (case-insensitive) one of the available clickable elements
- Always use lowercase for action names: `search` and `click`
- acts differently in each environment(try not to be same and repeat) and switch between environments properly
- think carefully before you buy
- if you go in a wrong direction(take no valid actions), you can switch environment(through tags) or go back to search 

Once you've finished your reasoning, you should choose admissible actions and present them within <parallel> </parallel> tags.

Your output must follow the rules above.'''


history_prompt = """You have already taken multiple actions in multiple parallel environments. Below are the most recent observations and the corresponding actions you took: {action_history}
"""


reason_prompt_para = """You are an expert agent operating in the Webshop Environment. 
Your task is to: {task_description}.
Your current observation is: {current_observation}
Your admissible actions are: 
[
{admissible_actions}
].

Now it's your turn to choose environments and take actions following the rules below:

1.There are {total_envs} environments available (indexed from 0 to {total_envs}-1), but you can only select up to {num_parallel} best environments to take actions each time. The actions of rest environments should be set to null.
2.You can explore 1 to {num_parallel} paths (indexed from 0 to {total_envs}-1), acts differently in each environment and switch between environments properly can shorten the shopping process.
3.You should first evaluate whether previous actions have taken effect based on the action history. This evaluation consists of 3 parts:
 1) whether the environment has changed
 2) whether the expected result has been achieved
 3) check current_observation(all {total_envs} environments should be considered) then choose a group of best environments(using <env_i> </env_i> tags) and take different actions(search, click or null)
4.Reason step-by-step about the current situation, and think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags. 
5.When all environments didn't change, check the content(clickable or not) and format(about actions and tags, from role:system in the beginning) of your output.
6.Invalid format(response and actions) and all null actions will fail your task, so check again before you finally response.
"""

# 新增了acts逻辑