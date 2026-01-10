"""
Streamlit Dashboard for Narrative Consistency System

Interactive UI for:
- Uploading novels
- Entering backstories
- Running the pipeline
- Viewing results and explanations
"""

import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core import (
    PathwayDocumentStore, HierarchicalNarrativeMemory,
    ClaimExtractor, ConstraintBuilder, MultiHopRetriever,
    CausalReasoningEngine, TemporalReasoningEngine,
    InconsistencyScorer, ConsistencyClassifier
)
import pandas as pd
import time


# Page config
st.set_page_config(
    page_title="Narrative Consistency Checker",
    page_icon="📚",
    layout="wide"
)

# Title
st.title("📚 Long-Context Narrative Consistency System")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
    max_hops = st.slider("Max Hops", 1, 5, 3)
    top_k = st.slider("Top-K Evidence", 3, 10, 5)
    threshold = st.slider("Inconsistency Threshold", 0.3, 0.7, 0.5, 0.05)
    
    st.markdown("---")
    st.markdown("### 🔬 Research Foundation")
    st.markdown("""
    - **Process reasoning**: Zhu et al. 2025
    - **Narrative gaps**: Zhang & Long 2024
    - **Causal inference**: Feder et al. 2022
    - **Temporal tracking**: Sun et al. 2013
    - **Long-doc modeling**: Lu et al. 2023
    """)

# Initialize session state
if 'document_store' not in st.session_state:
    st.session_state.document_store = PathwayDocumentStore(
        embedding_model=None, 
        chunk_size=chunk_size
    )
    st.session_state.novel_loaded = False
    st.session_state.results = None

# Main content
tab1, tab2, tab3 = st.tabs(["📖 Upload Novel", "🔍 Check Consistency", "📊 Results"])

# Tab 1: Upload Novel
with tab1:
    st.header("Upload Novel")
    st.markdown("Upload a novel text file (100k+ words recommended)")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a text file",
            type=['txt'],
            help="Upload the novel you want to check backstories against"
        )
        
        if uploaded_file:
            novel_text = uploaded_file.read().decode('utf-8')
            novel_name = uploaded_file.name
            
            st.success(f"✓ Loaded: {novel_name} ({len(novel_text):,} characters)")
            
            if st.button("🚀 Ingest Novel", type="primary"):
                with st.spinner("Ingesting novel... This may take a minute."):
                    progress_bar = st.progress(0)
                    
                    # Ingest
                    chunk_ids = st.session_state.document_store.ingest_novel(
                        novel_text=novel_text,
                        novel_id=novel_name,
                        metadata={'filename': novel_name}
                    )
                    progress_bar.progress(100)
                    
                    st.session_state.novel_loaded = True
                    st.session_state.novel_name = novel_name
                    
                    st.success(f"✓ Ingestion complete! Created {len(chunk_ids)} chunks")
                    
                    # Show stats
                    stats = st.session_state.document_store.get_statistics()
                    st.json(stats)
    
    with col2:
        if st.session_state.novel_loaded:
            st.info("📚 Novel loaded and ready!")
            st.metric("Novel", st.session_state.novel_name)
        else:
            st.warning("⚠️ No novel loaded yet")
        
        # Sample novel option
        st.markdown("---")
        if st.button("📝 Use Sample Novel"):
            sample_novel = """
            The Chronicles of Evermoor: A Gothic Mystery
            
            In the autumn of 1847, Elizabeth Hartwell arrived at Evermoor Manor...
            [Sample text would continue here]
            """
            st.session_state.document_store.ingest_novel(
                novel_text=sample_novel,
                novel_id="sample_evermoor",
                metadata={'title': 'Evermoor', 'type': 'sample'}
            )
            st.session_state.novel_loaded = True
            st.session_state.novel_name = "sample_evermoor"
            st.success("✓ Sample novel loaded!")

# Tab 2: Check Consistency
with tab2:
    st.header("Check Backstory Consistency")
    
    if not st.session_state.novel_loaded:
        st.warning("⚠️ Please upload a novel first (Tab 1)")
    else:
        st.success(f"✓ Using novel: {st.session_state.novel_name}")
        
        # Backstory input
        backstory = st.text_area(
            "Enter hypothetical backstory:",
            height=200,
            placeholder="Example: Before arriving at the manor, Elizabeth had lived in Paris for two years. She met Thomas during a ball in London in 1845...",
            help="Enter a backstory to check against the novel"
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col2:
            run_button = st.button("🔍 Run Analysis", type="primary", use_container_width=True)
        
        if run_button and backstory:
            st.markdown("---")
            st.subheader("🔬 Analysis in Progress")
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Extract claims
                status_text.text("1/7 Extracting claims...")
                progress_bar.progress(14)
                
                extractor = ClaimExtractor()
                claims = extractor.extract_claims(backstory)
                st.success(f"✓ Extracted {len(claims)} claims")
                time.sleep(0.3)
                
                # Step 2: Build memory
                status_text.text("2/7 Building narrative memory...")
                progress_bar.progress(28)
                
                memory = HierarchicalNarrativeMemory()
                chunks = []
                for chunk_id, doc in st.session_state.document_store.documents.items():
                    if st.session_state.document_store.chunk_to_doc.get(chunk_id) == st.session_state.novel_name:
                        chunks.append({
                            'chunk_id': chunk_id,
                            'text': doc.text,
                            'metadata': doc.metadata
                        })
                memory.extract_narrative_from_chunks(chunks, st.session_state.novel_name)
                st.success(f"✓ Built memory with {len(memory.scenes)} scenes")
                time.sleep(0.3)
                
                # Step 3: Build constraints
                status_text.text("3/7 Building constraint graph...")
                progress_bar.progress(42)
                
                builder = ConstraintBuilder()
                constraint_graph = builder.build_graph(claims)
                st.success(f"✓ Built graph with {len(constraint_graph.constraints)} constraints")
                time.sleep(0.3)
                
                # Step 4: Retrieve evidence
                status_text.text("4/7 Retrieving evidence...")
                progress_bar.progress(56)
                
                retriever = MultiHopRetriever(st.session_state.document_store, max_hops=max_hops)
                evidence_map = retriever.retrieve_for_claims(
                    claims=claims,
                    novel_id=st.session_state.novel_name,
                    top_k_per_claim=top_k
                )
                total_evidence = sum(len(ev) for ev in evidence_map.values())
                st.success(f"✓ Retrieved {total_evidence} evidence pieces")
                time.sleep(0.3)
                
                # Step 5: Reasoning
                status_text.text("5/7 Applying causal & temporal reasoning...")
                progress_bar.progress(70)
                
                causal_engine = CausalReasoningEngine(memory, constraint_graph)
                temporal_engine = TemporalReasoningEngine(memory, constraint_graph)
                
                temporal_engine.build_timeline(claims, evidence_map)
                temporal_conflicts = temporal_engine.check_temporal_consistency(claims, evidence_map)
                causal_conflicts = causal_engine.check_causal_consistency(claims, evidence_map)
                
                st.success(f"✓ Found {len(temporal_conflicts)} temporal + {len(causal_conflicts)} causal conflicts")
                time.sleep(0.3)
                
                # Step 6: Scoring
                status_text.text("6/7 Computing inconsistency score...")
                progress_bar.progress(84)
                
                scorer = InconsistencyScorer()
                score_result = scorer.score_backstory(
                    claims=claims,
                    evidence_map=evidence_map,
                    temporal_conflicts=temporal_conflicts,
                    causal_conflicts=causal_conflicts,
                    memory=memory
                )
                st.success(f"✓ Inconsistency score: {score_result['overall_inconsistency']:.3f}")
                time.sleep(0.3)
                
                # Step 7: Classification
                status_text.text("7/7 Making final classification...")
                progress_bar.progress(100)
                
                classifier = ConsistencyClassifier(threshold=threshold)
                classification = classifier.classify(
                    inconsistency_score=score_result['overall_inconsistency'],
                    temporal_conflicts=temporal_conflicts,
                    causal_conflicts=causal_conflicts,
                    evidence_map=evidence_map,
                    claims=claims
                )
                
                status_text.text("✓ Analysis complete!")
                time.sleep(0.5)
                
                # Store results
                st.session_state.results = {
                    'classification': classification,
                    'claims': claims,
                    'evidence_map': evidence_map,
                    'temporal_conflicts': temporal_conflicts,
                    'causal_conflicts': causal_conflicts,
                    'score_result': score_result
                }
                
                # Display result
                st.markdown("---")
                st.markdown("### 🎯 Result")
                
                if classification['prediction'] == 1:
                    st.success("### ✅ CONSISTENT")
                    st.markdown("The backstory is globally consistent with the novel.")
                else:
                    st.error("### ❌ INCONSISTENT")
                    st.markdown("The backstory contradicts the novel.")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Confidence", f"{classification['confidence']:.1%}")
                col2.metric("Inconsistency Score", f"{classification['inconsistency_score']:.3f}")
                col3.metric("Conflicts", classification['num_conflicts'])
                
                st.info(f"**Rationale:** {classification['rationale']}")
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                status_text.text("Analysis failed")

# Tab 3: Results
with tab3:
    st.header("Detailed Results")
    
    if st.session_state.results is None:
        st.info("ℹ️ No results yet. Run an analysis first (Tab 2)")
    else:
        results = st.session_state.results
        
        # Summary metrics
        st.subheader("📊 Summary Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric(
            "Prediction",
            "Consistent" if results['classification']['prediction'] == 1 else "Inconsistent",
            delta=None
        )
        col2.metric("Confidence", f"{results['classification']['confidence']:.1%}")
        col3.metric("Inconsistency", f"{results['classification']['inconsistency_score']:.3f}")
        col4.metric("Total Conflicts", results['classification']['num_conflicts'])
        
        st.markdown("---")
        
        # Claims
        st.subheader("📝 Extracted Claims")
        claims_data = []
        for claim in results['claims']:
            claims_data.append({
                'ID': claim.claim_id,
                'Type': claim.claim_type,
                'Text': claim.text[:80] + "..." if len(claim.text) > 80 else claim.text,
                'Entities': ', '.join(claim.entities[:3])
            })
        st.dataframe(pd.DataFrame(claims_data), use_container_width=True)
        
        st.markdown("---")
        
        # Evidence
        st.subheader("🔍 Evidence")
        total_evidence = sum(len(ev) for ev in results['evidence_map'].values())
        st.write(f"Total evidence pieces: {total_evidence}")
        
        # Show evidence for first claim
        if results['evidence_map']:
            first_claim_id = list(results['evidence_map'].keys())[0]
            with st.expander(f"Evidence for {first_claim_id}"):
                for i, ev in enumerate(results['evidence_map'][first_claim_id][:3], 1):
                    st.markdown(f"**{i}. Score: {ev.score:.3f}**")
                    st.text(ev.text[:200] + "...")
                    st.markdown("---")
        
        # Conflicts
        st.subheader("⚠️ Detected Conflicts")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Temporal Conflicts**")
            if results['temporal_conflicts']:
                for conflict in results['temporal_conflicts'][:3]:
                    with st.expander(f"{conflict.conflict_type} (severity: {conflict.severity:.2f})"):
                        st.write(conflict.description)
            else:
                st.success("No temporal conflicts")
        
        with col2:
            st.markdown("**Causal Conflicts**")
            if results['causal_conflicts']:
                for conflict in results['causal_conflicts'][:3]:
                    with st.expander(f"{conflict.conflict_type} (severity: {conflict.severity:.2f})"):
                        st.write(conflict.description)
            else:
                st.success("No causal conflicts")
        
        st.markdown("---")
        
        # Component scores
        st.subheader("🎯 Component Scores")
        
        if results['score_result']['claim_scores']:
            score_data = []
            for cs in results['score_result']['claim_scores'][:5]:
                score_data.append({
                    'Claim': cs['claim_id'],
                    'Overall': f"{cs['overall_inconsistency']:.3f}",
                    'Temporal': f"{cs['components']['temporal']:.3f}",
                    'Causal': f"{cs['components']['causal']:.3f}",
                    'Entity': f"{cs['components']['entity']:.3f}",
                    'Semantic': f"{cs['components']['semantic']:.3f}",
                    'Evidence': f"{cs['components']['evidence']:.3f}"
                })
            st.dataframe(pd.DataFrame(score_data), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Long-Context Narrative Consistency System | Built with Streamlit</p>
    <p>Research: Zhu et al. 2025 | Zhang & Long 2024 | Feder et al. 2022 | Sun et al. 2013</p>
</div>
""", unsafe_allow_html=True)
