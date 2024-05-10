import { Sentence } from './Sentence';

export class Paragraph {
  private sentences: Sentence[] = [];

  add(sentence: Sentence) {
    this.sentences.push(sentence);
    return this;
  }

  toString(): string {
    if (this.sentences.length == 0) {
      return '';
    }

    return this.sentences.join(' ');
  }
}
