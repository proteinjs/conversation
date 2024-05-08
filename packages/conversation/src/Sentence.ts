// Alina was here
export class Sentence {
  private lines: string[] = [];

  add(line: string) {
    this.lines.push(line);
    return this;
  }

  toString(): string {
    if (this.lines.length == 0) {
      return '';
    }

    let sentence = this.lines.join(', ');
    if (sentence.lastIndexOf('.') != sentence.length - 1) {
      sentence += '.';
    }

    return sentence;
  }
}
