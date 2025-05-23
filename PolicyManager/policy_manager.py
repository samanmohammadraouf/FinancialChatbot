class PolicyManager:
    def __init__(self, intents: list, intent_slot_mapping: dict, non_related_intent: str = None):
        self.intents = intents
        self.intent_slot_mapping = intent_slot_mapping
        self.current_intent = None 
        self.filled_slots = {} 
        self.remaining_slots = {}
        self.non_sense_intent = non_related_intent
        self.slot_ask_count = {}

    def set_non_sense_intent(self, non_sense_intent_label: str):
        self.non_sense_intent = non_sense_intent_label

    def is_intent_not_related(self, intent=None) -> bool:
        if intent is not None:
            return intent == self.non_sense_intent
        if self.current_intent == None:
            raise ValueError(f"Current Intent is not specified! Use set_intent to determine the current intent.")
        return self.current_intent == self.non_sense_intent

    def set_intent(self, detected_intent: str):
        if detected_intent not in self.intent_slot_mapping:
            raise ValueError(f"Unknown intent: {detected_intent}")
        
        if detected_intent != self.current_intent:
            self.current_intent = detected_intent
            self.filled_slots = {}
            self.remaining_slots = {slot: None for slot in self.intent_slot_mapping[detected_intent]}

    def get_slots_for_intent(self) -> list:
        if self.current_intent is None:
            raise ValueError("No intent has been set yet.")
        
        return self.intent_slot_mapping[self.current_intent]

    def update_slots(self, detected_slots: list) -> tuple:
        if self.current_intent is None:
            raise ValueError("No intent has been set yet.")

        current_slot = None
        current_value = []
        
        # Process detected slots
        for word, slot_tag in detected_slots:
            if slot_tag.startswith("b-"):
                # save the previous slot ...
                if current_slot and current_slot in self.remaining_slots:
                    self.filled_slots[current_slot] = " ".join(current_value)
                    if current_slot in self.remaining_slots:
                        del self.remaining_slots[current_slot]
                
                # Start new slot
                current_slot = slot_tag[2:]
                current_value = [word]
                
            elif slot_tag.startswith("i-"):
                slot_name = slot_tag[2:]
                if slot_name == current_slot:
                    # Continue current slot
                    current_value.append(word)
                else:
                    # Invalid I-tag without B-tag - treat as new slot
                    current_slot = None
                    current_value = []
                    
            else:  # O tag ... finalize any current slot
                if current_slot and current_slot in self.remaining_slots:
                    self.filled_slots[current_slot] = " ".join(current_value)
                    if current_slot in self.remaining_slots:
                        del self.remaining_slots[current_slot]
                    current_slot = None
                    current_value = []

        if current_slot and current_slot in self.remaining_slots:
            self.filled_slots[current_slot] = " ".join(current_value)
            if current_slot in self.remaining_slots:
                del self.remaining_slots[current_slot]

        return list(self.filled_slots.keys()), list(self.remaining_slots.keys())

    def update_slots_by_entity_detector(self, entity_tuples: list, slot_name: str) -> tuple:

        if self.current_intent is None:
            raise ValueError("No intent has been set yet.")

        if slot_name not in self.intent_slot_mapping[self.current_intent]:
            raise ValueError(f"Slot {slot_name} not valid for current intent {self.current_intent}")

        entity_words = [word for word, label in entity_tuples if label == 'E']
        
        if entity_words:
            slot_value = ' '.join(entity_words)
            self.filled_slots[slot_name] = slot_value
            if slot_name in self.remaining_slots:
                del self.remaining_slots[slot_name]

        return list(self.filled_slots.keys()), list(self.remaining_slots.keys())

    def update_slot_based_on_input(self, user_input: str, slot_name: str) -> tuple:
        if self.current_intent is None:
            raise ValueError("No intent has been set yet.")
        
        if slot_name not in self.intent_slot_mapping[self.current_intent]:
            raise ValueError(f"Slot {slot_name} not valid for current intent {self.current_intent}")
        
        self.filled_slots[slot_name] = user_input
        if slot_name in self.remaining_slots:
            del self.remaining_slots[slot_name]
            
        return list(self.filled_slots.keys()), list(self.remaining_slots.keys())

    def get_next_action(self) -> str:
        if self.current_intent is None:
            raise ValueError("No intent has been set yet.")
        
        if not self.remaining_slots:
            return "All slots filled for this intent."
        
        # Get and track the next slot
        next_slot = next(iter(self.remaining_slots.keys()))
        self.slot_ask_count[next_slot] = self.slot_ask_count.get(next_slot, 0) + 1
        return next_slot

    def get_remaining_slots(self) -> list:
        if self.current_intent is None:
            raise ValueError("No intent has been set yet.")
        
        return list(self.remaining_slots.keys())

    def get_current_state(self) -> dict:
        return {
            "current_intent": self.current_intent,
            "filled_slots": self.filled_slots,
            "remaining_slots": list(self.remaining_slots.keys())
        }

    def get_dialogue_state(self) -> dict:
        return {
            "intent": self.current_intent,
            "parameters": self.filled_slots.copy()
        }

    def get_slot_ask_counts(self) -> dict:
        return self.slot_ask_count.copy()