package org.lkpnotice.infra.metrics.seReport;

import java.util.HashMap;
import java.util.Map;

public class EmptyReport implements IReport {

    @Override
    public long stop() {
        return 0;
    }

    @Override
    public void logInternal(int level, boolean last) {
    }

    @Override
    public IReport getSubReport(String reportName) {
        return new EmptyReport();
    }

    @Override
    public void addEventCount(ReportItems eReportCount, int count) {

    }

    @Override
    public void start() {

    }

    @Override
    public void startIfPaused() {

    }

    @Override
    public long pause() {
        return 0;
    }

    @Override
    public long getElapsedTime() {
        return 0;
    }

    @Override
    public Map<String, IReport> getSubReports() {
        return new HashMap<String, IReport>();
    }

    @Override
    public String getName() {
        return "";
    }

    @Override
    public boolean isStarted() {
        return false;
    }

    @Override
    public void setStarted(boolean started) {

    }

    @Override
    public IStopWatch getStopWatch() {
        return new EmptyStopWatch();
    }

}