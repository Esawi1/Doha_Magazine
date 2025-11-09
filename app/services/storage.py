"""Storage service for chat history and feedback (Cosmos DB)"""
import uuid
from datetime import datetime
from typing import Any, List, Optional, Dict
from app.deps import get_cosmos_client, get_settings
from app.models import Message, Feedback, Source


class StorageService:
    """Service for storing chat messages and feedback"""
    
    def __init__(self):
        self.client = get_cosmos_client()
        self.settings = get_settings()
        self._messages_container = None
        self._feedback_container = None
        
        if self.client:
            self._init_containers()
    
    def _init_containers(self):
        """Initialize Cosmos DB containers"""
        try:
            database = self.client.get_database_client(self.settings.COSMOS_DB)
            
            # Create database if it doesn't exist
            try:
                self.client.create_database(self.settings.COSMOS_DB)
            except:
                pass  # Database already exists
            
            # Get or create messages container
            try:
                self._messages_container = database.create_container(
                    id=self.settings.COSMOS_CONTAINER_MESSAGES,
                    partition_key={"paths": ["/session_id"], "kind": "Hash"}
                )
            except:
                self._messages_container = database.get_container_client(
                    self.settings.COSMOS_CONTAINER_MESSAGES
                )
            
            # Get or create feedback container
            try:
                self._feedback_container = database.create_container(
                    id=self.settings.COSMOS_CONTAINER_FEEDBACK,
                    partition_key={"paths": ["/session_id"], "kind": "Hash"}
                )
            except:
                self._feedback_container = database.get_container_client(
                    self.settings.COSMOS_CONTAINER_FEEDBACK
                )
        
        except Exception as e:
            print(f"[!] Failed to initialize Cosmos DB: {str(e)}")
            self._messages_container = None
            self._feedback_container = None
    
    def save_message(self, session_id: str, role: str, text: str) -> str:
        """
        Save a message to storage as part of a conversation session.
        
        Args:
            session_id: Chat session ID
            role: 'user' or 'assistant'
            text: Message text
        
        Returns:
            Message ID
        """
        if not self._messages_container:
            print("[!] Cosmos DB not available, message not saved")
            return ""
        
        message_id = str(uuid.uuid4())
        current_time = datetime.utcnow().isoformat()
        
        # Create the new message (simplified - no sources, timestamps, or metadata)
        new_message = {
            "id": message_id,
            "role": role,
            "text": text
        }
        
        # Only add feedback field for assistant messages (no hardcoded null)
        # Feedback will be added when user actually provides it
        
        try:
            # Try to get existing session document
            try:
                existing_session = self._messages_container.read_item(
                    item=session_id,
                    partition_key=session_id
                )
                
                # Update existing session with new message
                if "messages" not in existing_session:
                    existing_session["messages"] = []
                
                existing_session["messages"].append(new_message)
                existing_session["message_count"] = len(existing_session["messages"])
                
                # Update the session document
                self._messages_container.upsert_item(body=existing_session)
                print(f"[+] Updated session {session_id} with new message")
                
                # Update feedback statistics if this is an assistant message
                if role == "assistant":
                    self.update_feedback_statistics(session_id)
                
            except Exception:
                # Session doesn't exist, create new session document
                session_data = {
                    "id": session_id,
                    "session_id": session_id,
                    "messages": [new_message],
                    "message_count": 1
                }
                
                self._messages_container.create_item(body=session_data)
                print(f"[+] Created new session {session_id} with first message")
            
            return message_id
            
        except Exception as e:
            print(f"[!] Failed to save message: {str(e)}")
            return ""
    
    def get_session_messages(self, session_id: str, limit: int = 50) -> List[Message]:
        """
        Get messages for a session.
        
        Args:
            session_id: Chat session ID
            limit: Maximum number of messages to return
        
        Returns:
            List of Message objects
        """
        if not self._messages_container:
            return []
        
        try:
            # Get the session document
            session_doc = self._messages_container.read_item(
                item=session_id,
                partition_key=session_id
            )
            
            # Extract messages from the session document
            messages = session_doc.get("messages", [])
            
            # Convert to Message objects and apply limit
            message_objects = []
            for msg in messages[-limit:]:  # Get the last 'limit' messages
                # Only include feedback for assistant messages
                feedback_value = msg.get("feedback") if msg["role"] == "assistant" else None
                message_objects.append(Message(
                    id=msg["id"],
                    role=msg["role"],
                    text=msg["text"],
                    feedback=feedback_value
                ))
            
            return message_objects
        
        except Exception as e:
            # If session doesn't exist yet, that's normal for new sessions
            if "NotFound" in str(e) or "does not exist" in str(e):
                return []  # Return empty list for new sessions
            else:
                print(f"[!] Failed to get messages: {str(e)}")
                return []
    
    def update_message_feedback(self, session_id: str, message_id: str, feedback: str) -> bool:
        """
        Update feedback for a specific message within a session.
        
        Args:
            session_id: Chat session ID
            message_id: Message ID to update
            feedback: 'up', 'down', or None
        
        Returns:
            True if successful, False otherwise
        """
        if not self._messages_container:
            print("[!] Cosmos DB not available, feedback not saved")
            return False
        
        try:
            print(f"[FEEDBACK_UPDATE] Starting feedback update for session={session_id}, message={message_id}, feedback={feedback}")
            
            # Get the session document
            session_doc = self._messages_container.read_item(
                item=session_id,
                partition_key=session_id
            )
            
            # Find and update the specific message
            messages = session_doc.get("messages", [])
            updated = False
            
            print(f"[FEEDBACK_UPDATE] Found {len(messages)} messages in session")
            
            for i, msg in enumerate[Any](messages):
                msg_id = msg.get('id', 'no-id')
                msg_role = msg.get('role', 'unknown')
                current_feedback = msg.get('feedback', 'no-field')
                
                print(f"[FEEDBACK_UPDATE] Message {i}: id={msg_id}, role={msg_role}, current_feedback={current_feedback}")
                
                if msg_id == message_id:
                    # Only update feedback for assistant messages
                    if msg_role == "assistant":
                        print(f"[FEEDBACK_UPDATE] Found target assistant message, updating feedback from {current_feedback} to {feedback}")
                        msg["feedback"] = feedback
                        updated = True
                        print(f"[FEEDBACK_UPDATE] Message updated in memory: {msg}")
                        break
                    else:
                        print(f"[FEEDBACK_UPDATE] Cannot update feedback for user message")
                        return False
            
            if updated:
                # Update the session document
                try:
                    print(f"[FEEDBACK_UPDATE] Upserting session document...")
                    self._messages_container.upsert_item(body=session_doc)
                    print(f"[FEEDBACK_UPDATE] Successfully updated feedback for message {message_id} in session {session_id}")
                    
                    # Verify the update by reading the document back
                    print(f"[FEEDBACK_UPDATE] Verifying update...")
                    verify_doc = self._messages_container.read_item(
                        item=session_id,
                        partition_key=session_id
                    )
                    verify_messages = verify_doc.get("messages", [])
                    for msg in verify_messages:
                        if msg["id"] == message_id:
                            print(f"[FEEDBACK_UPDATE] Verification: Message {message_id} now has feedback = {msg.get('feedback')}")
                            break
                    
                    return True
                except Exception as e:
                    print(f"[FEEDBACK_UPDATE] Failed to upsert session document: {str(e)}")
                    return False
            else:
                print(f"[FEEDBACK_UPDATE] Message {message_id} not found in session {session_id}")
                print(f"[FEEDBACK_UPDATE] Available message IDs: {[msg.get('id') for msg in messages]}")
                return False
                
        except Exception as e:
            # If session doesn't exist yet, that's normal for new sessions
            if "NotFound" in str(e) or "does not exist" in str(e):
                print(f"[FEEDBACK_UPDATE] Session {session_id} doesn't exist yet")
                return False
            else:
                print(f"[FEEDBACK_UPDATE] Failed to update message feedback: {str(e)}")
                return False

    def update_feedback_statistics(self, session_id: str) -> Dict:
        """
        Calculate and update feedback statistics for a session.
        
        Args:
            session_id: Session ID to calculate statistics for
            
        Returns:
            Dictionary with feedback statistics
        """
        if not self._messages_container:
            print("[!] Cosmos DB not available, cannot calculate statistics")
            return {}
        
        try:
            # Get the session document
            session_doc = self._messages_container.read_item(
                item=session_id,
                partition_key=session_id
            )
            
            messages = session_doc.get("messages", [])
            
            # Count feedback statistics
            positive_count = 0
            negative_count = 0
            null_count = 0
            total_assistant_messages = 0
            
            print(f"[*] Calculating feedback statistics for {len(messages)} messages")
            
            for i, msg in enumerate(messages):
                if msg.get("role") == "assistant":
                    total_assistant_messages += 1
                    feedback = msg.get("feedback")
                    has_feedback_field = "feedback" in msg
                    
                    print(f"[STATS] Message {i}: role={msg.get('role')}, feedback={feedback}, has_field={has_feedback_field}")
                    
                    if feedback == "up":
                        positive_count += 1
                        print(f"[STATS] Found positive feedback: {positive_count}")
                    elif feedback == "down":
                        negative_count += 1
                        print(f"[STATS] Found negative feedback: {negative_count}")
                    elif feedback is None and has_feedback_field:
                        # Only count as null if feedback field exists but is None
                        null_count += 1
                        print(f"[STATS] Found null feedback: {null_count}")
                    elif not has_feedback_field:
                        print(f"[STATS] Message has no feedback field yet (not counted as null)")
                    else:
                        # Handle any other feedback values
                        null_count += 1
                        print(f"[STATS] Found other feedback value '{feedback}': {null_count}")
            
            print(f"[*] Final counts - positive: {positive_count}, negative: {negative_count}, null: {null_count}, total: {total_assistant_messages}")
            
            # Calculate overall feedback
            if total_assistant_messages > 0:
                positive_ratio = positive_count / total_assistant_messages
                negative_ratio = negative_count / total_assistant_messages
                
                if positive_ratio > negative_ratio and positive_ratio > 0.3:
                    overall_feedback = "positive"
                elif negative_ratio > positive_ratio and negative_ratio > 0.3:
                    overall_feedback = "negative"
                else:
                    overall_feedback = "neutral"
            else:
                overall_feedback = "no_feedback"
            
            # Create statistics object
            statistics = {
                "positive": positive_count,
                "negative": negative_count,
                "null": null_count,
                "total_assistant_messages": total_assistant_messages,
                "positive_ratio": positive_count / total_assistant_messages if total_assistant_messages > 0 else 0,
                "negative_ratio": negative_count / total_assistant_messages if total_assistant_messages > 0 else 0,
                "overall_feedback": overall_feedback,
                "last_updated": datetime.utcnow().isoformat()
            }
            
            # Update session document with statistics
            session_doc["feedback_statistics"] = statistics
            self._messages_container.upsert_item(body=session_doc)
            
            print(f"[+] Updated feedback statistics for session {session_id}: {statistics}")
            return statistics
            
        except Exception as e:
            print(f"[!] Failed to calculate feedback statistics: {str(e)}")
            return {}

    def get_feedback_statistics(self, session_id: str) -> Dict:
        """
        Get feedback statistics for a session.
        
        Args:
            session_id: Session ID to get statistics for
            
        Returns:
            Dictionary with feedback statistics
        """
        if not self._messages_container:
            print("[!] Cosmos DB not available, cannot get statistics")
            return {}
        
        try:
            # Get the session document
            session_doc = self._messages_container.read_item(
                item=session_id,
                partition_key=session_id
            )
            
            return session_doc.get("feedback_statistics", {})
            
        except Exception as e:
            # If session doesn't exist yet, that's normal for new sessions
            if "NotFound" in str(e) or "does not exist" in str(e):
                return {}  # Return empty stats for new sessions
            else:
                print(f"[!] Failed to get feedback statistics: {str(e)}")
                return {}

    def force_update_statistics(self, session_id: str) -> Dict:
        """
        Force update feedback statistics for a session (useful for existing sessions).
        
        Args:
            session_id: Session ID to update statistics for
            
        Returns:
            Dictionary with updated feedback statistics
        """
        print(f"[*] FORCE_UPDATE: Force updating statistics for session {session_id}")
        return self.update_feedback_statistics(session_id)

    def recalculate_all_statistics(self, session_id: str) -> Dict:
        """
        Recalculate feedback statistics from scratch for a session.
        This ensures all feedbacks are counted correctly.
        
        Args:
            session_id: Session ID to recalculate statistics for
            
        Returns:
            Dictionary with recalculated feedback statistics
        """
        print(f"[*] RECALCULATE: Recalculating all statistics for session {session_id}")
        
        if not self._messages_container:
            print("[!] Cosmos DB not available, cannot recalculate statistics")
            return {}
        
        try:
            # Get the session document
            session_doc = self._messages_container.read_item(
                item=session_id,
                partition_key=session_id
            )
            
            messages = session_doc.get("messages", [])
            print(f"[*] RECALCULATE: Found {len(messages)} total messages")
            
            # Count feedback statistics from scratch
            positive_count = 0
            negative_count = 0
            null_count = 0
            total_assistant_messages = 0
            
            for i, msg in enumerate(messages):
                if msg.get("role") == "assistant":
                    total_assistant_messages += 1
                    feedback = msg.get("feedback")
                    print(f"[*] RECALCULATE: Message {i} (assistant): feedback={feedback}")
                    
                    if feedback == "up":
                        positive_count += 1
                        print(f"[+] RECALCULATE: Found positive feedback: {positive_count}")
                    elif feedback == "down":
                        negative_count += 1
                        print(f"[+] RECALCULATE: Found negative feedback: {negative_count}")
                    elif feedback is None:
                        # Only count as null if feedback field exists but is None
                        # Don't count messages that don't have feedback field at all
                        if "feedback" in msg:
                            null_count += 1
                            print(f"[+] RECALCULATE: Found null feedback: {null_count}")
                        else:
                            print(f"[*] RECALCULATE: Message has no feedback field yet (not counted as null)")
                    else:
                        # Handle any other feedback values
                        null_count += 1
                        print(f"[+] RECALCULATE: Found other feedback value '{feedback}': {null_count}")
            
            print(f"[*] RECALCULATE: Final counts - positive: {positive_count}, negative: {negative_count}, null: {null_count}, total: {total_assistant_messages}")
            
            # Calculate overall feedback
            if total_assistant_messages > 0:
                positive_ratio = positive_count / total_assistant_messages
                negative_ratio = negative_count / total_assistant_messages
                
                if positive_ratio > negative_ratio and positive_ratio > 0.3:
                    overall_feedback = "positive"
                elif negative_ratio > positive_ratio and negative_ratio > 0.3:
                    overall_feedback = "negative"
                else:
                    overall_feedback = "neutral"
            else:
                overall_feedback = "no_feedback"
            
            # Create statistics object
            statistics = {
                "positive": positive_count,
                "negative": negative_count,
                "null": null_count,
                "total_assistant_messages": total_assistant_messages,
                "positive_ratio": positive_count / total_assistant_messages if total_assistant_messages > 0 else 0,
                "negative_ratio": negative_count / total_assistant_messages if total_assistant_messages > 0 else 0,
                "overall_feedback": overall_feedback,
                "last_updated": datetime.utcnow().isoformat()
            }
            
            # Update session document with statistics
            session_doc["feedback_statistics"] = statistics
            self._messages_container.upsert_item(body=session_doc)
            
            print(f"[+] RECALCULATE: Updated statistics for session {session_id}: {statistics}")
            return statistics
            
        except Exception as e:
            print(f"[!] RECALCULATE: Failed to recalculate statistics: {str(e)}")
            return {}

    def save_feedback(self, session_id: str, message_id: str, 
                      rating: str, comment: Optional[str] = None) -> str:
        """
        Save user feedback and update statistics.
        
        Args:
            session_id: Chat session ID
            message_id: Message ID being rated
            rating: 'up' or 'down'
            comment: Optional comment (ignored in new structure)
        
        Returns:
            Message ID if successful, empty string otherwise
        """
        print(f"[*] SAVE_FEEDBACK: Starting feedback save for session={session_id}, message={message_id}, rating={rating}")
        
        # Update the specific message feedback
        success = self.update_message_feedback(session_id, message_id, rating)
        if success:
            print(f"[+] SAVE_FEEDBACK: Message feedback updated successfully, now updating statistics")
            # Update overall feedback statistics
            statistics = self.update_feedback_statistics(session_id)
            print(f"[+] SAVE_FEEDBACK: Statistics updated: {statistics}")
            return message_id  # Return message ID as feedback ID
        else:
            print(f"[!] SAVE_FEEDBACK: Failed to update message feedback")
            return ""
    
    def get_unanswered_queries(self, limit: int = 50) -> List[Dict]:
        """
        Get queries where the assistant couldn't provide good answers.
        
        Args:
            limit: Maximum number to return
        
        Returns:
            List of unanswered query dicts
        """
        if not self._messages_container:
            return []
        
        try:
            # Find user messages where assistant replied with low confidence
            # This is a simplified version - you'd need more sophisticated detection
            query = """
            SELECT c.id, c.session_id, c.text, c.created_utc, c.meta
            FROM c 
            WHERE c.role = 'user' 
            AND (c.meta.no_results = true OR c.meta.low_score = true)
            ORDER BY c.created_utc DESC
            OFFSET 0 LIMIT @limit
            """
            
            items = self._messages_container.query_items(
                query=query,
                parameters=[{"name": "@limit", "value": limit}],
                enable_cross_partition_query=True
            )
            
            return list(items)
        
        except Exception as e:
            print(f"[!] Failed to get unanswered queries: {str(e)}")
            return []


# Global storage service instance
_storage_service = None


def get_storage_service() -> StorageService:
    """Get or create storage service instance"""
    global _storage_service
    if _storage_service is None:
        _storage_service = StorageService()
    return _storage_service

