import { Moment } from '../opt/moment';
import { DateTimeColumn, StringColumn } from '../Columns';
import { Table } from '../Table';
import { SourceRecord, withSourceRecordColumns } from '../source/SourceRecord';

export interface Migration extends SourceRecord {
  description: string;
  status?: 'proposed'|'running'|'success'|'failure';
  failureMessage?: string;
  failureStack?: string;
  startTime?: Moment;
  endTime?: Moment;
  run: () => Promise<void>;
}

export class MigrationTable extends Table<Migration> {
  public name = 'migration';
	public columns = withSourceRecordColumns<Migration>({
    description: new StringColumn('description', {}, 4000),
    status: new StringColumn('status', { defaultValue: async () => 'proposed' }),
    failureMessage: new StringColumn('failure_message', {}, 4000),
    failureStack: new StringColumn('failure_stack', {}, 'MAX'),
    startTime: new DateTimeColumn('start_time'),
    endTime: new DateTimeColumn('end_time'),
	});
  public sourceRecordOptions = {
    doNotDeleteSourceRecordsFromDb: true,
  };
}