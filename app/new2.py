import openai
import json
import streamlit as st
from openai import OpenAI
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Expanded Chatbot Personas with more explicit annotation guidelines
chatbot_personas = {
    "Emily": "You are a 45-year-old Caucasian female social scientist with a Ph.D. in Health Communication. You are known for a meticulous, precise approach to data annotation. When analyzing text, follow these strict guidelines: 1) Always refer back to the exact codebook definitions. 2) If a text entry doesn't perfectly match a category, choose the closest match. 3) Prioritize objective, factual interpretation over emotional reading. 4) Be consistent in applying categories across different texts.",
    "Michael": "You are a 38-year-old Hispanic male social scientist with a Ph.D. in Sociology. You are known for an empathetic, context-sensitive approach to data annotation. When analyzing text, follow these guidelines: 1) Consider the broader emotional and social context. 2) Look for nuanced interpretations that capture the underlying narrative. 3) If a text entry spans multiple categories, explain your reasoning. 4) Prioritize understanding the human experience behind the text."
}

def get_chatbot_discussion_and_update_codebook(text_entries, annotations, codebook):
    prompt = f"Emily and Michael have independently annotated the following text entries with their annotations: \n\n"
    disagreement_entries = []

    
    for i, entry in enumerate(text_entries):
        emily_annotation = annotations[entry]['Emily']
        michael_annotation = annotations[entry]['Michael']
        
        if emily_annotation != michael_annotation:
            disagreement_entries.append(entry)
            prompt += f"Text {i+1}: {entry}\n\n"
            prompt += f"Emily's Annotation: {emily_annotation}\n"
            prompt += f"Michael's Annotation: {michael_annotation}\n\n"
    
    if not disagreement_entries:
        st.write("No disagreements found in annotations.")
        return codebook
    
    prompt += (
        "Based on the differences in their annotations, they will discuss how to refine the codebook rules "
        "to make them clearer and more specific. The codebook rules can still only contain 4 labels/categories "
        "that have labels 1-4. There should not be any additional labels. Do not include updates to the narrator perpspective, "
        "the updates should only be to the descriptions/refinement of narrative events."
        " Provide a transcript of their discussion and propose updated rules for the codebook."
        "\n\nReturn only the updated rules as JSON."
    )

    response = openai.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[
            {"role": "system", "content": "You are facilitating a discussion between two personas (Emily and Michael) to refine codebook annotations."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=800
    )
    
    discussion_and_updates = response.choices[0].message.content.strip()
    
    # Display the discussion
    st.write("Discussion Transcript:")
    st.write(discussion_and_updates)
    
    # Extract JSON for updated codebook
    import json
    try:
        json_start_index = discussion_and_updates.find("{")
        json_data = discussion_and_updates[json_start_index:]
        updated_rules = json.loads(json_data)
        
        # Update the codebook with new rules
        for category, updates in updated_rules.items():
            if category in codebook:
                codebook[category].update(updates)
            else:
                codebook[category] = updates
        
        st.success("Codebook updated successfully!")
        return codebook
    except json.JSONDecodeError:
        st.error("Failed to parse updated rules. Please check the discussion output for manual updates.")
        
        return codebook
    
# Performance Tracking Class
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score

def parse_multi_label(x):
    """
    Parse multi-label ground truth that might be:
    - Single integer (e.g., 3)
    - Comma-separated string of integers (e.g., '3,4')
    - List of integers
    
    Returns a list of integers
    """
    if isinstance(x, (int, np.integer)):
        return [x]
    elif isinstance(x, str):
        # Split comma-separated values and convert to integers
        return [int(val.strip()) for val in x.split(',')]
    elif isinstance(x, list):
        return [int(val) for val in x]
    else:
        return []

def multi_label_metrics(y_true, y_pred):
    """
    Calculate multi-label performance metrics
    
    Parameters:
    y_true (list): Ground truth multi-label annotations
    y_pred (list): Predicted multi-label annotations
    
    Returns:
    dict: Performance metrics
    """
    # Ensure all inputs are lists of lists
    y_true = [parse_multi_label(y) for y in y_true]
    y_pred = [parse_multi_label(y) for y in y_pred]
    
    # Use MultiLabelBinarizer to transform labels
    mlb = MultiLabelBinarizer()
    y_true_bin = mlb.fit_transform(y_true)
    y_pred_bin = mlb.transform(y_pred)
    
    # Calculate metrics
    metrics = {
        'exact_match_accuracy': accuracy_score(y_true_bin, y_pred_bin),
      
    }
    
    return metrics

# Modify the AnnotationTracker to use multi-label metrics
class AnnotationTracker:
    def __init__(self):
        self.rounds = []
        self.performance_metrics = []
        self.disagreement_counts = []
        self.agreement_rates = []  # Add this to track agreement rates

    def calculate_agreement_rate(self, emily_annotations, michael_annotations):
        """
        Calculate the agreement rate between Emily and Michael's annotations.
        Returns tuple of (ne_agreement_rate, np_agreement_rate)
        """
        ne_agreements = 0
        np_agreements = 0
        total_entries = len(emily_annotations)
        
        for emily_ann, michael_ann in zip(emily_annotations, michael_annotations):
            # Compare NE annotations
            emily_ne = set(parse_multi_label(emily_ann.get('NE', [])))
            michael_ne = set(parse_multi_label(michael_ann.get('NE', [])))
            if emily_ne == michael_ne:
                ne_agreements += 1
                
            # Compare NP annotations
            emily_np = set(parse_multi_label(emily_ann.get('NP', [])))
            michael_np = set(parse_multi_label(michael_ann.get('NP', [])))
            if emily_np == michael_np:
                np_agreements += 1
        
        ne_agreement_rate = ne_agreements / total_entries if total_entries > 0 else 0
        np_agreement_rate = np_agreements / total_entries if total_entries > 0 else 0
        
        return ne_agreement_rate, np_agreement_rate

    def add_round_results(self, emily_annotations, michael_annotations, ground_truth):
        # Calculate agreement rates first
        ne_agreement_rate, np_agreement_rate = self.calculate_agreement_rate(emily_annotations, michael_annotations)
        self.agreement_rates.append((ne_agreement_rate, np_agreement_rate))
        
        # Extract NE and NP from annotations
        emily_ne = [
            ann.get('NE', []) if isinstance(ann.get('NE'), list) 
            else [ann.get('NE')] if ann.get('NE') 
            else [] 
            for ann in emily_annotations
        ]
        michael_ne = [
            ann.get('NE', []) if isinstance(ann.get('NE'), list) 
            else [ann.get('NE')] if ann.get('NE') 
            else [] 
            for ann in michael_annotations
        ]
        
        emily_np = [
            ann.get('NP', []) if isinstance(ann.get('NP'), list) 
            else [ann.get('NP')] if ann.get('NP') 
            else [] 
            for ann in emily_annotations
        ]
        michael_np = [
            ann.get('NP', []) if isinstance(ann.get('NP'), list) 
            else [ann.get('NP')] if ann.get('NP') 
            else [] 
            for ann in michael_annotations
        ]
        
        # Ground truth processing
        ground_truth_ne = ground_truth[:, 0]  # NE column
        ground_truth_np = ground_truth[:, 1]  # NP column
        
        # Calculate multi-label metrics
        ne_metrics = multi_label_metrics(ground_truth_ne, emily_ne)
        np_metrics = multi_label_metrics(ground_truth_np, emily_np)
        
        # Count disagreements (different from ground truth)
        ne_disagreements = sum(
            1 for e, g in zip(emily_ne, ground_truth_ne) 
            if set(parse_multi_label(e)) != set(parse_multi_label(g))
        )
        np_disagreements = sum(
            1 for e, g in zip(emily_np, ground_truth_np) 
            if set(parse_multi_label(e)) != set(parse_multi_label(g))
        )
        
        # Store results
        round_result = {
            'NE_metrics': ne_metrics,
            'NP_metrics': np_metrics,
            'NE_disagreements': ne_disagreements,
            'NP_disagreements': np_disagreements,
            'NE_agreement_rate': ne_agreement_rate,
            'NP_agreement_rate': np_agreement_rate
        }
        
        self.rounds.append(round_result)
        self.performance_metrics.append((ne_metrics, np_metrics))
        self.disagreement_counts.append((ne_disagreements, np_disagreements))
        
        return round_result

    def plot_performance(self):
        """
        Create a comprehensive visualization of all metrics
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Plot Exact Match Accuracy
        ne_accuracy = [r[0]['exact_match_accuracy'] for r in self.performance_metrics]
        np_accuracy = [r[1]['exact_match_accuracy'] for r in self.performance_metrics]
        
        axes[0, 0].plot(ne_accuracy, label='Narrative Event Accuracy', marker='o')
        axes[0, 0].plot(np_accuracy, label='Narrator Perspective Accuracy', marker='o')
        axes[0, 0].set_title('Exact Match Accuracy')
        axes[0, 0].set_xlabel('Training Round')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. Plot Disagreements
        ne_disagreements = [r[0] for r in self.disagreement_counts]
        np_disagreements = [r[1] for r in self.disagreement_counts]
        
        axes[0, 1].plot(ne_disagreements, label='Narrative Event Disagreements', marker='o')
        axes[0, 1].plot(np_disagreements, label='Narrator Perspective Disagreements', marker='o')
        axes[0, 1].set_title('Annotation Disagreements')
        axes[0, 1].set_xlabel('Training Round')
        axes[0, 1].set_ylabel('Number of Disagreements')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. Plot Agreement Rates
        ne_agreement_rates = [r[0] for r in self.agreement_rates]
        np_agreement_rates = [r[1] for r in self.agreement_rates]
        
        axes[1, 0].plot(ne_agreement_rates, label='NE Agreement Rate', marker='o', color='green')
        axes[1, 0].plot(np_agreement_rates, label='NP Agreement Rate', marker='o', color='purple')
        axes[1, 0].set_title('Inter-Annotator Agreement Rates')
        axes[1, 0].set_xlabel('Training Round')
        axes[1, 0].set_ylabel('Agreement Rate')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 4. Plot Combined Metrics
        axes[1, 1].plot(ne_agreement_rates, label='NE Agreement', marker='o', color='green')
        axes[1, 1].plot(ne_accuracy, label='NE Accuracy', marker='s', color='blue')
        axes[1, 1].set_title('Agreement vs Accuracy (NE)')
        axes[1, 1].set_xlabel('Training Round')
        axes[1, 1].set_ylabel('Rate')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        return fig
    
def get_annotation_from_api(entry, persona, codebook):
    
    prompt = f"Persona: {persona}\n\n" + \
         "ANNOTATION INSTRUCTIONS:\n" + \
         "1. Carefully read the text entry\n" + \
         "2. Identify the Narrative Event (NE) category\n" + \
         "3. Identify the Narrator Perspective (NP) category\n" + \
         "4. Provide a clear rationale for your choice\n\n" + \
         "CODEBOOK:\n" + \
         "Narrative Events (NE):\n" + \
         "\n".join([f"{k}: {v['label']} - {v['description']}" for k, v in codebook["Narrative Event(s) related to breast cancer 'NE'"].items()]) + \
         "\n\nNarrator Perspective (NP):\n" + \
         "\n".join([f"{k}: {v['label']} - {v['description']}" for k, v in codebook["Narrator perspective 'NP'"].items()]) + \
         f"\n\nText Entry:\n{entry}\n\n" + \
         "Response Format (JSON):\n" + \
         "{\n" + \
         '  "NE": "<selected NE category number>",\n' + \
         '  "NP": "<selected NP category number>",\n' + \
         '  "rationale": "<explanation of your choice>"\n' + \
         "}"

    
    response = openai.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[
            {"role": "system", "content": "You are an expert in content analysis, carefully annotating text entries."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,  # Reduced temperature for more consistent results
        response_format={"type": "json_object"},
        max_tokens=500
    )
    
    return response.choices[0].message.content.strip()
# Main Streamlit App
def main():
    codebook = {
    "Narrative Event(s) related to breast cancer 'NE'": {
        "1": {
            "label": "Prevention",
            "description": "Related to actions or events focused on preventing breast cancer."
        },
        "2": {
            "label": "Detection, diagnosis",
            "description": "Events related to the detection and diagnosis of breast cancer."
        },
        "3": {
            "label": "Treatment",
            "description": "Events involving treatment of breast cancer.",
            "subcategories": {
                "3.1": "Receiving treatment (e.g., getting the IV chemo, lying in the hospital bed)",
                "3.2": "Treatment effects (e.g., bald head, flat chest, wearing a head wrap)",
                "3.3": "Treatment milestone or completion (e.g., ringing the chemo bell, showing radiation therapy completion certificate)"
            }
        },
        "4": {
            "label": "Survivorship",
            "description": "Events related to life after cancer treatment, including remission, recurrence, or death.",
            "subcategories": {
                "4.1": "Complete remission/cancer free; recurrence; a second cancer; and death",
                "4.2": "Fundraising, any prosocial or philanthropic activities"
            }
        }
    },
    "Narrator perspective 'NP'": {
        "1": {
            "label": "Breast cancer survivor",
            "description": "The narrator is someone who has survived breast cancer."
        },
        "2": {
            "label": "Breast cancer survivor’s family or friends",
            "description": "The narrator is a family member or friend of a breast cancer survivor."
        },
        "3": {
            "label": "Mixed (i.e., survivor + family or friends)",
            "description": "The narrator is a combination of the breast cancer survivor and their family or friends."
        },
        "4": {
            "label": "Journalists/news media",
            "description": "The narrator is a journalist or part of the news media."
        },
        "5": {
            "label": "Breast cancer organization",
            "description": "The narrator is from a breast cancer organization."
        }
    }
}

    st.title("Iterative Content Analysis Training")
    
    # Initialize session state for tracking
    if 'annotation_tracker' not in st.session_state:
        st.session_state.annotation_tracker = AnnotationTracker()
    
    # Load dataset
    st.header("Data Upload")
    uploaded_data = st.file_uploader("Upload the Dataset (Excel Sheet)", type=["xlsx"])
    
    if uploaded_data is not None:
        # Read data
        data_sheet = pd.read_excel(uploaded_data)
        
        # Implement multi-round training
        num_rounds = 3
        num_entries_per_round = 10
        
        for round_num in range(1, num_rounds + 1):
            st.header(f"Training Round {round_num}")
            
            # Select entries for this round
            start_idx = (round_num - 1) * num_entries_per_round
            end_idx = start_idx + num_entries_per_round
            text_entries = data_sheet.loc[start_idx:end_idx-1, "content"].to_numpy()
            
            # Annotations dictionary
            annotations = {entry: {"Emily": None, "Michael": None} for entry in text_entries}
            
            # Independent Annotation Phase
            st.subheader("Independent Annotation")
            for entry in text_entries:
                st.markdown(f"**Text Entry:** *{entry}*")
                
                # Annotate with both personas
                emily_result = get_annotation_from_api(entry, chatbot_personas["Emily"], codebook)
                michael_result = get_annotation_from_api(entry, chatbot_personas["Michael"], codebook)
                
                # Parse JSON results
                try:
                    emily_annotation = json.loads(emily_result)
                    michael_annotation = json.loads(michael_result)
                    
                   #displays annotations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Emily's Annotation:**")
                        st.json({
                            "NE": emily_annotation['NE'],
                            "NP": emily_annotation['NP'],
                            "Rationale": emily_annotation.get('rationale', 'No rationale provided')
                        })
                    
                    with col2:
                        st.markdown("**Michael's Annotation:**")
                        st.json({
                            "NE": michael_annotation['NE'],
                            "NP": michael_annotation['NP'],
                            "Rationale": michael_annotation.get('rationale', 'No rationale provided')
                        })
                    
                    # Highlight disagreements
                    if emily_annotation['NE'] != michael_annotation['NE'] or emily_annotation['NP'] != michael_annotation['NP']:
                        st.warning("⚠️ Disagreement detected between Emily and Michael's annotations!")
                    
                    # Store annotations
                    annotations[entry]["Emily"] = emily_annotation
                    annotations[entry]["Michael"] = michael_annotation
                    
                except json.JSONDecodeError:
                    st.error(f"Failed to parse annotations for entry: {entry}")
                    continue
            
            # Performance Tracking
            st.subheader("Performance Metrics")
            # Assuming ground truth is available in the dataset
            ground_truth = data_sheet.loc[start_idx:end_idx-1, ["NE", "NP"]].to_numpy()
           
                 # Performance Tracking
   
            round_performance = st.session_state.annotation_tracker.add_round_results(
                [ann["Emily"] for ann in annotations.values()],
                [ann["Michael"] for ann in annotations.values()],
                ground_truth
            )
        
             # Display round performance
            st.write("Round Performance:")
            st.write({
                "NE Agreement Rate": f"{round_performance['NE_agreement_rate']:.2%}",
                "NP Agreement Rate": f"{round_performance['NP_agreement_rate']:.2%}",
                "NE Accuracy": f"{round_performance['NE_metrics']['exact_match_accuracy']:.2%}",
                "NP Accuracy": f"{round_performance['NP_metrics']['exact_match_accuracy']:.2%}"
            })
            # Chatbot Discussion and Codebook Refinement
            st.subheader("Chatbot Discussion")
            codebook = get_chatbot_discussion_and_update_codebook(text_entries, annotations, codebook)
        
        # Final Performance Visualization
        st.header("Overall Performance")
        performance_fig = st.session_state.annotation_tracker.plot_performance()
        st.pyplot(performance_fig)

if __name__ == "__main__":
    main()