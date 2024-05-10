export type Question = {
  text: string;
  optional?: boolean;
};

export type ConversationTemplate = {
  name: string;
  keywords: string[];
  description: string;
  questions: Question[];
  instructions: () => Promise<string[]>;
};
