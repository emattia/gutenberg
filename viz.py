import re
import torch
from typing import Optional, List, Dict, Any
from torchtune.modules.tokenizers import ModelTokenizer

def display_prompt(
    prompt, 
    answer, 
    tokenizer
):
    """
    Display prompt in HTML container for Jupyter and in cards of Metaflow tasks.

    Args:
        prompt: The tensor of prompt tokens
        answer: A string with the answer to the prompt
        tokenizer: The tokenizer to decode responses
    """
    html_output = """
        <style>
            .prompt-box {
                margin: 20px 0;
                padding: 15px;
                border: 1px solid #C4C7AC;
                border-radius: 8px;
                background-color: #F0EBE5;
                color: #4A4A67;
                font-family: 'Courier New', monospace;
                font-size: 14px;
                line-height: 1.6;
                white-space: pre-wrap;
                word-wrap: break-word;
            }
        </style>
    """

    ### PROCESS PROMPTS ###
    if hasattr(prompt, 'shape'):
        prompt_decoded = tokenizer.decode(
            prompt[0].tolist() 
            if len(prompt.shape) > 0 and prompt.shape[0] > 0 else prompt.tolist()
        )
    else:
        prompt_decoded = str(prompt)

    ### SHOW PROMPT ###
    formatted_prompt = prompt_decoded.replace("<", "&lt;").replace(">", "&gt;")
    formatted_prompt = formatted_prompt.replace("\n", "<br>")

    html_output += '<div class="prompt-box">'
    html_output += f'<div style="background-color: #F0EBE5; font-size: 16px; font-weight: bold;">Prompt</div>{formatted_prompt}'
    html_output += "<br>"
    html_output += f"<strong>Ground Truth Answer: {answer}</strong>"
    html_output += '</div>'

    return html_output

def display_responses(
    responses: torch.Tensor,
    tokenizer: ModelTokenizer,
    grpo_size: int,
    advantages: Optional[torch.Tensor] = None,
    rewards: Optional[torch.Tensor] = None,
    successes: Optional[torch.Tensor] = None,
    details: Optional[List[List[Dict[str, Any]]]] = None,
    show_n: Optional[int] = None
) -> str:
    """
    Display responses with rewards, advantages, and detailed diagnostics in a visually appealing format.
    
    Args:
        responses: Tensor of token IDs
        tokenizer: Tokenizer for decoding responses
        grpo_size: Size of the policy optimization group
        advantages: Optional tensor of advantages
        rewards: Optional tensor of rewards
        successes: Optional tensor of successes
        details: Optional list of reward calculation details
        show_n: Optional maximum number of responses to display
        
    Returns:
        HTML string for displaying the responses
    """
    batch_size = responses.shape[0]
    
    # Helper function to safely get values from tensors with different shapes
    def get_item_value(tensor, batch_idx, group_idx):
        if tensor is None:
            return None
        
        if tensor.dim() == 1:
            # Handle 1D tensor [grpo_size]
            return tensor[group_idx].item()
        else:
            # Handle 2D tensor [batch_size, grpo_size]
            return tensor[batch_idx][group_idx].item()
    
    html_output = """
    <style>
        .response-container {
            margin: 20px 0;
            border: 1px solid #C4C7AC;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            font-family: 'Courier New', monospace;
            max-width: 100%;
        }
        .response-header {
            background-color: #F0EBE5;
            padding: 10px 15px;
            font-size: 16px;
            font-weight: bold;
            border-bottom: 1px solid #C4C7AC;
            color: #4A4A67;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .response-body {
            background-color: #ffffff;
            color: #4A4A67;
            padding: 15px;
            white-space: pre-wrap;
            word-wrap: break-word;
            line-height: 1.6;
            font-size: 14px;
        }
        .think-tag {
            color: #BE6A1A;
            font-weight: bold;
        }
        .answer-tag {
            color: #2C6846;
            font-weight: bold;
        }
        .answer-era-tag {
            color: #2C6846;
            font-weight: bold;
        }
        .answer-date-tag {
            color: #2C6846;
            font-weight: bold;
        }
        .metrics-container {
            background-color: #F0EBE5;
            border-top: 1px solid #C4C7AC;
            padding: 10px 15px;
        }
        .metric-label {
            color: #4A4A67;
        }
        .metric-score {
            font-family: monospace;
            font-weight: bold;
            padding: 2px 8px;
            border-radius: 4px;
            display: inline-block;
            margin-right: 8px;
        }
        .score-high {
            background-color: #D3EFE0;
            color: #177350;
        }
        .score-medium {
            background-color: #FCF1D6;
            color: #BE6A1A;
        }
        .score-low {
            background-color: #FAD9D8;
            color: #C5393A;
        }
        .success-badge {
            background-color: #177350;
            color: white;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        }
        .failure-badge {
            background-color: #C5393A;
            color: white;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        }
        .metrics-toggle {
            cursor: pointer;
            color: #3F7DC9;
            text-decoration: underline;
            font-size: 12px;
            margin-top: 5px;
            display: inline-block;
            font-weight: bold;
        }
        .details-container {
            display: none;
            margin-top: 10px;
            border-top: 1px solid #C4C7AC;
            padding-top: 10px;
        }
        
        /* Reward details styling */
        .reward-component {
            margin-bottom: 10px;
            padding: 8px;
            border-radius: 4px;
            background-color: #f8f9fa;
        }
        .component-name {
            font-weight: bold;
            color: #4A4A67;
        }
        .component-value {
            font-family: monospace;
            padding: 2px 4px;
            border-radius: 3px;
        }
        .component-value-positive {
            background-color: #D3EFE0;
            color: #177350;
        }
        .component-value-negative {
            background-color: #FAD9D8;
            color: #C5393A;
        }
        .component-reason {
            font-size: 0.9em;
            color: #555;
            margin-top: 4px;
        }
        .check-success {
            color: #177350;
            font-weight: bold;
        }
        .check-fail {
            color: #C5393A;
            font-weight: bold;
        }
        .batch-header {
            margin: 30px 0 10px 0;
            padding: 5px 10px;
            background-color: #E5E7D9;
            border-left: 4px solid #4A4A67;
            color: #4A4A67;
            font-size: 18px;
            font-weight: bold;
        }
    </style>
    
    <script>
    function toggleDetails(batchIdx, groupIdx) {
        var detailsId = 'details-' + batchIdx + '-' + groupIdx;
        var details = document.getElementById(detailsId);
        var buttonId = 'toggle-' + batchIdx + '-' + groupIdx;
        var toggleBtn = document.getElementById(buttonId);
        
        if (details) {
            if (details.style.display === 'none' || details.style.display === '') {
                details.style.display = 'block';
                if (toggleBtn) toggleBtn.innerText = 'Hide Details';
            } else {
                details.style.display = 'none';
                if (toggleBtn) toggleBtn.innerText = 'Show Details';
            }
        }
    }
    </script>
    """
    
    if show_n is not None:
        grpo_size = min(grpo_size, show_n)
    
    for b in range(batch_size):
        html_output += f'<div class="batch-header">Batch #{b+1}</div>'
        
        for g in range(grpo_size):
            # Decode the response
            response_text = tokenizer.decode(responses[b, g].tolist())
            
            # Determine if this response succeeded
            success_value = get_item_value(successes, b, g) if successes is not None else None
            is_successful = success_value == 1.0 if success_value is not None else None
            
            # Get reward and advantage if available
            reward_value = get_item_value(rewards, b, g) if rewards is not None else None
            advantage_value = get_item_value(advantages, b, g) if advantages is not None else None
            
            # Start response container
            html_output += f'<div class="response-container">'
            
            # Response header
            html_output += f'<div class="response-header">'
            html_output += f'<div>Response #{g+1}</div>'
            
            # Add success/fail badge if available
            if is_successful is not None:
                if is_successful:
                    html_output += f'<div class="success-badge">SUCCESS</div>'
                else:
                    html_output += f'<div class="failure-badge">FAIL</div>'
            
            html_output += '</div>'  # End response header
            
            # Response body with tag highlighting
            html_output += f'<div class="response-body">'
            
            # Escape HTML but preserve line breaks
            escaped_text = html.escape(response_text).replace('\n', '<br>')
            
            # Highlight tags
            escaped_text = re.sub(
                r'&lt;think&gt;(.+?)&lt;/think&gt;',
                r'<span class="think-tag">&lt;think&gt;</span>\1<span class="think-tag">&lt;/think&gt;</span>',
                escaped_text,
                flags=re.DOTALL
            )
            
            escaped_text = re.sub(
                r'&lt;answer_era&gt;(.+?)&lt;/answer_era&gt;',
                r'<span class="answer-era-tag">&lt;answer_era&gt;</span>\1<span class="answer-era-tag">&lt;/answer_era&gt;</span>',
                escaped_text
            )
            
            escaped_text = re.sub(
                r'&lt;answer_date&gt;(.+?)&lt;/answer_date&gt;',
                r'<span class="answer-date-tag">&lt;answer_date&gt;</span>\1<span class="answer-date-tag">&lt;/answer_date&gt;</span>',
                escaped_text
            )
            
            html_output += escaped_text
            html_output += '</div>'  # End response body
            
            # Metrics container
            if reward_value is not None or advantage_value is not None:
                html_output += f'<div class="metrics-container">'
                
                # Determine score class based on reward value
                score_class = "score-high" if reward_value and reward_value >= 80 else \
                              "score-medium" if reward_value and reward_value >= 30 else \
                              "score-low"
                
                # Display reward
                if reward_value is not None:
                    html_output += f'<div><strong class="metric-label">Reward:</strong> <span class="metric-score {score_class}">{reward_value:.1f}</span></div>'
                
                # Display advantage
                if advantage_value is not None:
                    adv_class = "score-high" if advantage_value > 0 else "score-low"
                    html_output += f'<div><strong class="metric-label">Advantage:</strong> <span class="metric-score {adv_class}">{advantage_value:.1f}</span></div>'
                
                # Add details toggle if available
                if details is not None and b < len(details) and g < len(details[b]):
                    html_output += f'<a id="toggle-{b}-{g}" class="metrics-toggle" onclick="toggleDetails({b}, {g})">Show Details</a>'
                    html_output += f'<div id="details-{b}-{g}" class="details-container">'
                    
                    # Format reward details
                    detail_data = details[b][g]
                    
                    # Ground truth
                    html_output += f'<div style="margin-bottom: 15px;">'
                    html_output += f'<div><strong>Ground Truth:</strong> Era=\'{detail_data["ground_truth"]["era"]}\', Date=\'{detail_data["ground_truth"]["date"]}\'</div>'
                    html_output += f'</div>'
                    
                    # Extracted tags
                    html_output += f'<div style="margin-bottom: 15px;">'
                    html_output += f'<div><strong>Extracted Tags:</strong></div>'
                    html_output += f'<ul style="margin-top: 5px; padding-left: 20px;">'
                    html_output += f'<li>Think: {len(detail_data["extracted_tags"]["think"])} tag(s)</li>'
                    html_output += f'<li>Era: {len(detail_data["extracted_tags"]["answer_era"])} tag(s)</li>'
                    html_output += f'<li>Date: {len(detail_data["extracted_tags"]["answer_date"])} tag(s)</li>'
                    html_output += f'</ul>'
                    html_output += f'</div>'
                    
                    # Format analysis
                    html_output += f'<div style="margin-bottom: 15px;">'
                    html_output += f'<div><strong>Format Analysis:</strong></div>'
                    
                    if detail_data['format_analysis'].get('has_outside_text', False):
                        outside_text = html.escape(detail_data['format_analysis']['outside_text'])
                        html_output += f'<div class="check-fail">❌ Text outside tags: "{outside_text}"</div>'
                    else:
                        html_output += f'<div class="check-success">✓ No text outside tags</div>'
                    
                    html_output += f'</div>'
                    
                    # Content analysis
                    if 'content_analysis' in detail_data:
                        html_output += f'<div style="margin-bottom: 15px;">'
                        html_output += f'<div><strong>Content Analysis:</strong></div>'
                        
                        # Era analysis
                        if 'era' in detail_data['content_analysis']:
                            provided_eras = ', '.join([f"'{era}'" for era in detail_data['content_analysis']['era']['provided']])
                            html_output += f'<div style="margin-top: 5px;"><strong>Era provided:</strong> {provided_eras}</div>'
                            
                            match_status = ""
                            if detail_data['content_analysis'].get('era_match', {}).get('exact_match', False):
                                match_status = f'<span class="check-success">✓ Exact match</span>'
                            elif detail_data['content_analysis'].get('era_match', {}).get('partial_match', False):
                                match_status = f'<span style="color: #BE6A1A; font-weight: bold;">~ Partial match</span>'
                            else:
                                match_status = f'<span class="check-fail">❌ No match</span>'
                            
                            html_output += f'<div><strong>Era match:</strong> {match_status}</div>'
                        
                        # Date analysis
                        if 'date' in detail_data['content_analysis']:
                            provided_dates = ', '.join([f"'{date}'" for date in detail_data['content_analysis']['date']['provided']])
                            html_output += f'<div style="margin-top: 5px;"><strong>Date provided:</strong> {provided_dates}</div>'
                            
                            if 'best_match' in detail_data['content_analysis']['date']:
                                best = detail_data['content_analysis']['date']['best_match']
                                diff_class = "check-success" if best['difference'] <= 20 else \
                                             ("color: #BE6A1A; font-weight: bold;" if best['difference'] <= 50 else "check-fail")
                                
                                html_output += f'<div><strong>Best date:</strong> {best["value"]} <span style="{diff_class}">(diff: {best["difference"]} years)</span></div>'
                        
                        html_output += f'</div>'
                    
                    # Reward components table
                    html_output += f'<div style="margin-bottom: 15px;">'
                    html_output += f'<div><strong>Reward Components:</strong></div>'
                    html_output += f'<div style="margin-top: 10px;">'
                    
                    for component in detail_data['reward_components']:
                        component_name = component['component']
                        value = component['value']
                        reason = component['reason']
                        
                        # Determine CSS class based on value
                        try:
                            value_float = float(str(value).replace("(overwrites previous)", ""))
                            value_class = "component-value-positive" if value_float > 0 else "component-value-negative"
                        except:
                            value_class = ""
                        
                        html_output += f'<div class="reward-component">'
                        html_output += f'<div><span class="component-name">{component_name}:</span> <span class="component-value {value_class}">{value}</span></div>'
                        html_output += f'<div class="component-reason">{reason}</div>'
                        html_output += f'</div>'
                    
                    html_output += f'</div>'
                    html_output += f'</div>'
                    
                    # Success criteria
                    if 'success_criteria' in detail_data:
                        criteria = detail_data['success_criteria']
                        html_output += f'<div style="margin-bottom: 15px;">'
                        html_output += f'<div><strong>Success Criteria:</strong></div>'
                        html_output += f'<ul style="margin-top: 5px; padding-left: 20px;">'
                        
                        html_output += f'<li><span class="{"check-success" if criteria["perfect_format"] else "check-fail"}">{("✓" if criteria["perfect_format"] else "❌")} Format perfect</span></li>'
                        html_output += f'<li><span class="{"check-success" if criteria["correct_era"] else "check-fail"}">{("✓" if criteria["correct_era"] else "❌")} Era correct</span></li>'
                        html_output += f'<li><span class="{"check-success" if criteria["correct_date"] else "check-fail"}">{("✓" if criteria["correct_date"] else "❌")} Date correct</span></li>'
                        
                        html_output += f'</ul>'
                        html_output += f'</div>'
                    
                    # Total reward summary
                    html_output += f'<div style="margin-top: 15px; padding-top: 10px; border-top: 1px solid #C4C7AC;">'
                    html_output += f'<div><strong>Total Reward:</strong> <span class="{"score-high" if detail_data["total_reward"] > 0 else "score-low"}">{detail_data["total_reward"]}</span></div>'
                    html_output += f'<div><strong>Success:</strong> <span class="{"score-high" if detail_data["success"] > 0 else "score-low"}">{detail_data["success"]}</span></div>'
                    html_output += f'</div>'
                    
                    html_output += f'</div>'  # End details container
                
                html_output += f'</div>'  # End metrics container
            
            html_output += f'</div>'  # End response container
    
    return html_output


# def display_responses_old(responses, tokenizer, grpo_size, advantages=None, rewards=None, successes=None, component_details=None, show_n:int=None):

#     """
#     Display prompt in HTML container for Jupyter and in cards of Metaflow tasks.
    
#     Args:
#         prompt: The tensor of prompt tokens
#         responses: The tensor of response tokens
#         tokenizer: The tokenizer to decode responses
#         grpo_size: Number of responses to print
#         advantages: Tensor of advantage values with shape [batch_size, grpo_size] or [grpo_size]
#         rewards: Tensor of reward values with shape [batch_size, grpo_size] or [grpo_size]
#         successes: Tensor of success indicators with shape [batch_size, grpo_size] or [grpo_size]
#         component_details: Optional list of component-level reward details
#         show_n: Optional maximum number of responses to display
    
#     Returns:
#         str that can be passed to IPython HTML display, embedded in Metaflow card or any other UI, of responses with optional reward metrics.
#     """
#     import re

#     # Handle tensor dimensions consistently
#     def get_item_value(tensor, index):
#         """Extract scalar value from tensor accounting for different dimensions."""
#         if tensor is None:
#             return None
        
#         if tensor.dim() == 1:
#             # Handle 1D tensor [grpo_size]
#             return tensor[index].item()
#         else:
#             # Handle 2D tensor [batch_size, grpo_size]
#             return tensor[0][index].item()  # Use first batch
    
#     html_output = """
#         <style>
#             .response-container {
#                 margin: 20px 0;
#                 border: 1px solid #C4C7AC;
#                 border-radius: 8px;
#                 overflow: hidden;
#                 box-shadow: 0 2px 5px rgba(0,0,0,0.05);
#                 font-family: 'Courier New', monospace;
#                 max-width: 100%;
#             }
#             .response-header {
#                 background-color: #F0EBE5;
#                 padding: 10px 15px;
#                 font-size: 16px;
#                 font-weight: bold;
#                 border-bottom: 1px solid #C4C7AC;
#                 color: #4A4A67;
#                 display: flex;
#                 justify-content: space-between;
#                 align-items: center;
#             }
#             .response-body {
#                 background-color: #ffffff;
#                 color: #4A4A67;
#                 padding: 15px;
#                 white-space: pre-wrap;
#                 word-wrap: break-word;
#                 line-height: 1.6;
#                 font-size: 14px;
#             }
#             .think-tag {
#                 color: #BE6A1A;
#                 font-weight: bold;
#             }
#             .answer-tag {
#                 color: #2C6846;
#                 font-weight: bold;
#             }
#             .answer-era-tag {
#                 color: #2C6846;
#                 font-weight: bold;
#             }
#             .answer-date-tag {
#                 color: #2C6846;
#                 font-weight: bold;
#             }
#             .calculation {
#                 color: #2E5CA7;
#                 background-color: #E9F0FA;
#                 padding: 2px 4px;
#                 border-radius: 2px;
#             }
#             .math {
#                 font-family: 'Courier New', monospace;
#                 background-color: #F0EBE5;
#                 padding: 0 3px;
#                 border-radius: 3px;
#             }
#             .metric-label {
#                 color: #4A4A67;
#             }
#             .metrics-container {
#                 background-color: #F0EBE5;
#                 border-top: 1px solid #C4C7AC;
#                 padding: 10px 15px;
#             }
#             .metric-score {
#                 font-family: monospace;
#                 font-weight: bold;
#                 padding: 2px 8px;
#                 border-radius: 4px;
#                 display: inline-block;
#                 margin-right: 8px;
#             }
#             .score-high {
#                 background-color: #D3EFE0;
#                 color: #177350;
#             }
#             .score-medium {
#                 background-color: #FCF1D6;
#                 color: #BE6A1A;
#             }
#             .score-low {
#                 background-color: #FAD9D8;
#                 color: #C5393A;
#             }
#             .success-badge {
#                 background-color: #177350;
#                 color: white;
#                 padding: 3px 8px;
#                 border-radius: 4px;
#                 font-size: 12px;
#                 font-weight: bold;
#             }
#             .failure-badge {
#                 background-color: #4A4A67;
#                 color: white;
#                 padding: 3px 8px;
#                 border-radius: 4px;
#                 font-size: 12px;
#                 font-weight: bold;
#             }
#             .component-table {
#                 width: 100%;
#                 border-collapse: collapse;
#                 margin-top: 10px;
#                 font-size: 13px;
#             }
#             .component-table th {
#                 background-color: #F0EBE5;
#                 text-align: left;
#                 padding: 6px 10px;
#                 color: #4A4A67;
#                 font-weight: bold;
#             }
#             .component-table td {
#                 border-top: 1px solid #C4C7AC;
#                 padding: 6px 10px;
#                 color: #4A4A67;
#             }
#             .component-progress {
#                 height: 8px;
#                 width: 100%;
#                 background-color: #E5E7D9;
#                 border-radius: 4px;
#                 overflow: hidden;
#             }
#             .component-progress-bar {
#                 height: 100%;
#                 background-color: #6B9BD0;
#             }
#             .metrics-toggle {
#                 cursor: pointer;
#                 color: #3F7DC9;
#                 text-decoration: underline;
#                 font-size: 12px;
#                 margin-top: 5px;
#                 display: inline-block;
#                 font-weight: bold;
#             }
#             .details-container {
#                 display: none;
#                 margin-top: 10px;
#                 border-top: 1px solid #C4C7AC;
#                 padding-top: 10px;
#             }
#     </style>
#     <script>
#     function toggleDetails(id) {
#         var details = document.getElementById('details-' + id);
#         if (details.style.display === 'none' || details.style.display === '') {
#             details.style.display = 'block';
#         } else {
#             details.style.display = 'none';
#         }
#     }
#     </script>
#     """
    
#     has_rewards = rewards is not None and successes is not None
#     has_details = component_details is not None
    
#     ### PROCESS RESPONSES ###
#     responses_decoded = []
#     for i in range(grpo_size):
#         try:
#             decoded = tokenizer.decode(responses[:, i, :].tolist()[0])
#             responses_decoded.append(decoded)
#         except:
#             responses_decoded.append("Sample response")
    
#     if show_n:
#         responses_decoded = responses_decoded[:show_n]

#     for i, response in enumerate(responses_decoded):
#         html_output += f'<div class="response-container">'
#         ### START HEADER ###
#         html_output += f'<div class="response-header">'
#         html_output += f'<div>Response #{i+1}</div>'
#         if has_rewards:
#             reward = get_item_value(rewards, i)
#             success = get_item_value(successes, i)
#             if success > 0.5:
#                 html_output += f'<div class="success-badge">SUCCESS</div>'
#             else:
#                 html_output += f'<div class="failure-badge">FAIL</div>'
#         html_output += '</div>'
#         ### START RESPONSE BODY ### 
#         html_output += f'<div class="response-body">'
#         response = response.replace("\n", "<br>")
#         # Escape HTML tags but keep <br> tags.
#         response = response.replace("<", "&lt;").replace(">", "&gt;")
#         response = response.replace("&lt;br&gt;", "<br>")
#         ### HIGHLIGHT THINK TAGS ###
#         response = re.sub(
#             r'&lt;/think&gt;', 
#             '<span class="think-tag">&lt;/think&gt;</span>', 
#             response
#         )
#         response = re.sub(
#             r'&lt;think&gt;', 
#             '<span class="think-tag">&lt;think&gt;</span>', 
#             response
#         )
#         ### HIGHLIGHT STANDARD ANSWER TAG ###
#         response = re.sub(
#             r'&lt;/answer&gt;', 
#             '<span class="answer-tag">&lt;/answer&gt;</span>', 
#             response
#         )
#         response = re.sub(
#             r'&lt;answer&gt;', 
#             '<span class="answer-tag">&lt;answer&gt;</span>', 
#             response
#         )
#         ### HIGHLIGHT ERA ANSWER TAG ###
#         response = re.sub(
#             r'&lt;/answer_era&gt;', 
#             '<span class="answer-era-tag">&lt;/answer_era&gt;</span>', 
#             response
#         )
#         response = re.sub(
#             r'&lt;answer_era&gt;', 
#             '<span class="answer-era-tag">&lt;answer_era&gt;</span>', 
#             response
#         )
#         ### HIGHLIGHT DATE ANSWER TAG ###
#         response = re.sub(
#             r'&lt;/answer_date&gt;', 
#             '<span class="answer-date-tag">&lt;/answer_date&gt;</span>', 
#             response
#         )
#         response = re.sub(
#             r'&lt;answer_date&gt;', 
#             '<span class="answer-date-tag">&lt;answer_date&gt;</span>', 
#             response
#         )
#         # HIGHLIGHT CALCULATIONS with $<<...>>$ pattern ###
#         response = re.sub(
#             r'\$&lt;&lt;(.+?)&gt;&gt;\$', 
#             r'<span class="calculation">$&lt;&lt;\1&gt;&gt;$</span>', 
#             response
#         )
#         ### FORMAT MATH EXPRESSIONS ###
#         response = re.sub(
#             r'(\d+[*/+-]\d+(?:[*/+-]\d+)*)', 
#             r'<span class="math">\1</span>', 
#             response
#         )
#         html_output += response
#         html_output += '</div>' 
#         ### START METRICS ###
#         if has_rewards:
#             html_output += f'<div class="metrics-container">'
#             reward = get_item_value(rewards, i)
#             advantage = get_item_value(advantages, i) if advantages is not None else None
#             score_class = "score-high" if reward >= 80 else "score-medium" if reward >= 30 else "score-low"
#             html_output += f'<div><strong class="metric-label">Reward:</strong> <span class="metric-score {score_class}">{reward:.1f}</span></div>'
#             if advantage is not None:
#                 html_output += f'<div><strong class="metric-label">Advantage:</strong> <span class="metric-score {score_class}">{advantage:.1f}</span></div>'
#             if has_details and i < len(component_details):
#                 details = component_details[i]
#                 html_output += f'<a class="metrics-toggle" onclick="toggleDetails({i})">Show component details</a>'
#                 html_output += f'<div id="details-{i}" class="details-container">'
#                 ### TABLE OF COMPONENTS ###
#                 html_output += '<table class="component-table">'
#                 html_output += '<tr><th style="text-align: left;">Component</th><th style="text-align: left;">Score</th><th style="text-align: left;">Contribution</th></tr>'
#                 for comp_name, result in details['component_results'].items():
#                     raw_score = result.raw_score
#                     weighted_score = result.weighted_score
#                     max_possible = result.max_possible
#                     is_satisfied = result.is_fully_satisfied
#                     # For progress bar.
#                     percentage = (weighted_score / max_possible) * 100 if max_possible > 0 else 0
#                     # Format row in table. Another SO to Claude.
#                     status_color = "#2b8a3e" if is_satisfied else "#e67700"
#                     html_output += f'<tr>'
#                     html_output += f'<td style="color: {status_color}; font-weight: {("bold" if is_satisfied else "normal")}">{comp_name}</td>'
#                     max_raw = max_possible/result.weighted_score*raw_score if raw_score > 0 and result.weighted_score > 0 else max_possible
#                     html_output += f'<td>{raw_score:.1f} / {max_raw:.1f}</td>'
#                     html_output += f'<td>'
#                     html_output += f'<div class="component-progress">'
#                     html_output += f'<div class="component-progress-bar" style="width: {min(percentage, 100)}%; background-color: {status_color if is_satisfied else "#4dabf7"};"></div>'
#                     html_output += f'</div>'
#                     html_output += f'</td>'
#                     html_output += f'</tr>'
#                 html_output += '</table>'
#                 # Show component by component 
#                 for comp_name, result in details['component_results'].items():
#                     if hasattr(result, 'debug_info') and result.debug_info:
#                         html_output += f'<div style="margin-top: 8px; font-size: 12px; color: #495057;">'
#                         html_output += f'<strong>{comp_name} details:</strong> '            
#                         # Format debug info
#                         debug_parts = []
#                         for key, value in result.debug_info.items():
#                             if isinstance(value, bool):
#                                 icon = "✓" if value else "✗"
#                                 color = "#2b8a3e" if value else "#c92a2a"
#                                 debug_parts.append(f'<span style="color: {color}">{key}: {icon}</span>')
#                             else:
#                                 debug_parts.append(f'{key}: {value}')
#                         html_output += ', '.join(debug_parts)
#                         html_output += '</div>'
#                 html_output += '</div>'  # Close details container
#             html_output += '</div>'  # Close metrics container
#         html_output += '</div>'  # Close response container
    
#     return html_output